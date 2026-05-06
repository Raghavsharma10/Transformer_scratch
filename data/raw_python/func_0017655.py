def setup_and_run_bwakit(job, uuid, url, rg_line, config, paired_url=None):
    """
    Downloads and runs bwakit for BAM or FASTQ files

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param str uuid: Unique sample identifier
    :param str url: FASTQ or BAM file URL. BAM alignment URL must have .bam extension.
    :param Namespace config: Input parameters and shared FileStoreIDs
        Requires the following config attributes:
        config.genome_fasta         FilesStoreID for reference genome fasta file
        config.genome_fai           FilesStoreID for reference genome fasta index file
        config.cores                Number of cores for each job
        config.trim                 If True, trim adapters using bwakit
        config.amb                  FileStoreID for BWA index file prefix.amb
        config.ann                  FileStoreID for BWA index file prefix.ann
        config.bwt                  FileStoreID for BWA index file prefix.bwt
        config.pac                  FileStoreID for BWA index file prefix.pac
        config.sa                   FileStoreID for BWA index file prefix.sa
        config.alt                  FileStoreID for alternate contigs file or None
    :param str|None paired_url: URL to paired FASTQ
    :param str|None rg_line: Read group line (i.e. @RG\tID:foo\tSM:bar)
    :return: BAM FileStoreID
    :rtype: str
    """
    bwa_config = deepcopy(config)
    bwa_config.uuid = uuid
    bwa_config.rg_line = rg_line

    # bwa_alignment uses a different naming convention
    bwa_config.ref = config.genome_fasta
    bwa_config.fai = config.genome_fai

    # Determine if sample is a FASTQ or BAM file using the file extension
    basename, ext = os.path.splitext(url)
    ext = ext.lower()
    if ext == '.gz':
        _, ext = os.path.splitext(basename)
        ext = ext.lower()

    # The pipeline currently supports FASTQ and BAM files
    require(ext in ['.fq', '.fastq', '.bam'],
            'Please use .fq or .bam file extensions:\n%s' % url)

    # Download fastq files
    samples = []
    input1 = job.addChildJobFn(download_url_job,
                               url,
                               name='file1',
                               s3_key_path=config.ssec,
                               disk=config.file_size)

    samples.append(input1.rv())

    # If the extension is for a BAM  file, then configure bwakit to realign the BAM file.
    if ext == '.bam':
        bwa_config.bam = input1.rv()
    else:
        bwa_config.r1 = input1.rv()

    # Download the paired FASTQ URL
    if paired_url:
        input2 = job.addChildJobFn(download_url_job,
                                   paired_url,
                                   name='file2',
                                   s3_key_path=config.ssec,
                                   disk=config.file_size)
        samples.append(input2.rv())
        bwa_config.r2 = input2.rv()

    # The bwakit disk requirement depends on the size of the input files and the index
    # Take the sum of the input files and scale it by a factor of 4
    bwa_index_size = sum([getattr(config, index_file).size
                          for index_file in ['amb', 'ann', 'bwt', 'pac', 'sa', 'alt']
                          if getattr(config, index_file, None) is not None])

    bwakit_disk = PromisedRequirement(lambda lst, index_size:
                                      int(4 * sum(x.size for x in lst) + index_size),
                                      samples,
                                      bwa_index_size)

    return job.addFollowOnJobFn(run_bwakit,
                                bwa_config,
                                sort=False,         # BAM files are sorted later in the pipeline
                                trim=config.trim,
                                cores=config.cores,
                                disk=bwakit_disk).rv()