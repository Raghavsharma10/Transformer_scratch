def gatk_germline_pipeline(job, samples, config):
    """
    Runs the GATK best practices pipeline for germline SNP and INDEL discovery.

    Steps in Pipeline
    0: Generate and preprocess BAM
        - Uploads processed BAM to output directory
    1: Call Variants using HaplotypeCaller
        - Uploads GVCF
    2: Genotype VCF
        - Uploads VCF
    3: Filter Variants using either "hard filters" or VQSR
        - Uploads filtered VCF

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param list[GermlineSample] samples: List of GermlineSample namedtuples
    :param Namespace config: Input parameters and reference FileStoreIDs
        Requires the following config attributes:
        config.genome_fasta         FilesStoreID for reference genome fasta file
        config.genome_fai           FilesStoreID for reference genome fasta index file
        config.genome_dict          FilesStoreID for reference genome sequence dictionary file
        config.cores                Number of cores for each job
        config.xmx                  Java heap size in bytes
        config.suffix               Suffix added to output filename
        config.output_dir           URL or local path to output directory
        config.ssec                 Path to key file for SSE-C encryption
        config.joint_genotype       If True, then joint genotype and filter cohort
        config.hc_output            URL or local path to HaplotypeCaller output for testing
    :return: Dictionary of filtered VCF FileStoreIDs
    :rtype: dict
    """
    require(len(samples) > 0, 'No samples were provided!')

    # Get total size of genome reference files. This is used for configuring disk size.
    genome_ref_size = config.genome_fasta.size + config.genome_fai.size + config.genome_dict.size

    # 0: Generate processed BAM and BAI files for each sample
    # group preprocessing and variant calling steps in empty Job instance
    group_bam_jobs = Job()
    gvcfs = {}
    for sample in samples:
        # 0: Generate processed BAM and BAI files for each sample
        get_bam = group_bam_jobs.addChildJobFn(prepare_bam,
                                               sample.uuid,
                                               sample.url,
                                               config,
                                               paired_url=sample.paired_url,
                                               rg_line=sample.rg_line)

        # 1: Generate per sample gvcfs {uuid: gvcf_id}
        # The HaplotypeCaller disk requirement depends on the input bam, bai, the genome reference
        # files, and the output GVCF file. The output GVCF is smaller than the input BAM file.
        hc_disk = PromisedRequirement(lambda bam, bai, ref_size:
                                      2 * bam.size + bai.size + ref_size,
                                      get_bam.rv(0),
                                      get_bam.rv(1),
                                      genome_ref_size)

        get_gvcf = get_bam.addFollowOnJobFn(gatk_haplotype_caller,
                                            get_bam.rv(0),
                                            get_bam.rv(1),
                                            config.genome_fasta, config.genome_fai, config.genome_dict,
                                            annotations=config.annotations,
                                            cores=config.cores,
                                            disk=hc_disk,
                                            memory=config.xmx,
                                            hc_output=config.hc_output)
        # Store cohort GVCFs in dictionary
        gvcfs[sample.uuid] = get_gvcf.rv()

        # Upload individual sample GVCF before genotyping to a sample specific output directory
        vqsr_name = '{}{}.g.vcf'.format(sample.uuid, config.suffix)
        get_gvcf.addChildJobFn(output_file_job,
                               vqsr_name,
                               get_gvcf.rv(),
                               os.path.join(config.output_dir, sample.uuid),
                               s3_key_path=config.ssec,
                               disk=PromisedRequirement(lambda x: x.size, get_gvcf.rv()))

    # VQSR requires many variants in order to train a decent model. GATK recommends a minimum of
    # 30 exomes or one large WGS sample:
    # https://software.broadinstitute.org/gatk/documentation/article?id=3225

    filtered_vcfs = {}
    if config.joint_genotype:
        # Need to configure joint genotype in a separate function to resolve promises
        filtered_vcfs = group_bam_jobs.addFollowOnJobFn(joint_genotype_and_filter,
                                                        gvcfs,
                                                        config).rv()

    # If not joint genotyping, then iterate over cohort and genotype and filter individually.
    else:
        for uuid, gvcf_id in gvcfs.iteritems():
            filtered_vcfs[uuid] = group_bam_jobs.addFollowOnJobFn(genotype_and_filter,
                                                                  {uuid: gvcf_id},
                                                                  config).rv()

    job.addChild(group_bam_jobs)
    return filtered_vcfs