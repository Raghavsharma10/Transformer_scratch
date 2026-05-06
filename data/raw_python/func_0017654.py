def prepare_bam(job, uuid, url, config, paired_url=None, rg_line=None):
    """
    Prepares BAM file for Toil germline pipeline.

    Steps in pipeline
    0: Download and align BAM or FASTQ sample
    1: Sort BAM
    2: Index BAM
    3: Run GATK preprocessing pipeline (Optional)
        - Uploads preprocessed BAM to output directory

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param str uuid: Unique identifier for the sample
    :param str url: URL or local path to BAM file or FASTQs
    :param Namespace config: Configuration options for pipeline
        Requires the following config attributes:
        config.genome_fasta         FilesStoreID for reference genome fasta file
        config.genome_fai           FilesStoreID for reference genome fasta index file
        config.genome_dict          FilesStoreID for reference genome sequence dictionary file
        config.g1k_indel            FileStoreID for 1000G INDEL resource file
        config.mills                FileStoreID for Mills resource file
        config.dbsnp                FileStoreID for dbSNP resource file
        config.suffix               Suffix added to output filename
        config.output_dir           URL or local path to output directory
        config.ssec                 Path to key file for SSE-C encryption
        config.cores                Number of cores for each job
        config.xmx                  Java heap size in bytes
    :param str|None paired_url: URL or local path to paired FASTQ file, default is None
    :param str|None rg_line: RG line for BWA alignment (i.e. @RG\tID:foo\tSM:bar), default is None
    :return: BAM and BAI FileStoreIDs
    :rtype: tuple
    """
    # 0: Align FASTQ or realign BAM
    if config.run_bwa:
        get_bam = job.wrapJobFn(setup_and_run_bwakit,
                                uuid,
                                url,
                                rg_line,
                                config,
                                paired_url=paired_url).encapsulate()

    # 0: Download BAM
    elif '.bam' in url.lower():
        job.fileStore.logToMaster("Downloading BAM: %s" % uuid)
        get_bam = job.wrapJobFn(download_url_job,
                                url,
                                name='toil.bam',
                                s3_key_path=config.ssec,
                                disk=config.file_size).encapsulate()
    else:
        raise ValueError('Could not generate BAM file for %s\n'
                         'Provide a FASTQ URL and set run-bwa or '
                         'provide a BAM URL that includes .bam extension.' % uuid)

    # 1: Sort BAM file if necessary
    # Realigning BAM file shuffles read order
    if config.sorted and not config.run_bwa:
        sorted_bam = get_bam

    else:
        # The samtools sort disk requirement depends on the input bam, the tmp files, and the
        # sorted output bam.
        sorted_bam_disk = PromisedRequirement(lambda bam: 3 * bam.size, get_bam.rv())
        sorted_bam = get_bam.addChildJobFn(run_samtools_sort,
                                           get_bam.rv(),
                                           cores=config.cores,
                                           disk=sorted_bam_disk)

    # 2: Index BAM
    # The samtools index disk requirement depends on the input bam and the output bam index
    index_bam_disk = PromisedRequirement(lambda bam: bam.size, sorted_bam.rv())
    index_bam = job.wrapJobFn(run_samtools_index, sorted_bam.rv(), disk=index_bam_disk)

    job.addChild(get_bam)
    sorted_bam.addChild(index_bam)

    if config.preprocess:
        preprocess = job.wrapJobFn(run_gatk_preprocessing,
                                   sorted_bam.rv(),
                                   index_bam.rv(),
                                   config.genome_fasta,
                                   config.genome_dict,
                                   config.genome_fai,
                                   config.g1k_indel,
                                   config.mills,
                                   config.dbsnp,
                                   memory=config.xmx,
                                   cores=config.cores).encapsulate()
        sorted_bam.addChild(preprocess)
        index_bam.addChild(preprocess)

        # Update output BAM promises
        output_bam_promise = preprocess.rv(0)
        output_bai_promise = preprocess.rv(1)

        # Save processed BAM
        output_dir = os.path.join(config.output_dir, uuid)
        filename = '{}.preprocessed{}.bam'.format(uuid, config.suffix)
        output_bam = job.wrapJobFn(output_file_job,
                                   filename,
                                   preprocess.rv(0),
                                   output_dir,
                                   s3_key_path=config.ssec)
        preprocess.addChild(output_bam)

    else:
        output_bam_promise = sorted_bam.rv()
        output_bai_promise = index_bam.rv()

    return output_bam_promise, output_bai_promise