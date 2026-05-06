def genotype_and_filter(job, gvcfs, config):
    """
    Genotypes one or more GVCF files and runs either the VQSR or hard filtering pipeline. Uploads the genotyped VCF file
    to the config output directory.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param dict gvcfs: Dictionary of GVCFs {Sample ID: FileStoreID}
    :param Namespace config: Input parameters and shared FileStoreIDs
        Requires the following config attributes:
        config.genome_fasta         FilesStoreID for reference genome fasta file
        config.genome_fai           FilesStoreID for reference genome fasta index file
        config.genome_dict          FilesStoreID for reference genome sequence dictionary file
        config.suffix               Suffix added to output filename
        config.output_dir           URL or local path to output directory
        config.ssec                 Path to key file for SSE-C encryption
        config.cores                Number of cores for each job
        config.xmx                  Java heap size in bytes
        config.unsafe_mode          If True, then run GATK tools in UNSAFE mode
    :return: FileStoreID for genotyped and filtered VCF file
    :rtype: str
    """
    # Get the total size of the genome reference
    genome_ref_size = config.genome_fasta.size + config.genome_fai.size + config.genome_dict.size

    # GenotypeGVCF disk requirement depends on the input GVCF, the genome reference files, and
    # the output VCF file. The output VCF is smaller than the input GVCF.
    genotype_gvcf_disk = PromisedRequirement(lambda gvcf_ids, ref_size:
                                             2 * sum(gvcf_.size for gvcf_ in gvcf_ids) + ref_size,
                                             gvcfs.values(),
                                             genome_ref_size)

    genotype_gvcf = job.addChildJobFn(gatk_genotype_gvcfs,
                                      gvcfs,
                                      config.genome_fasta,
                                      config.genome_fai,
                                      config.genome_dict,
                                      annotations=config.annotations,
                                      unsafe_mode=config.unsafe_mode,
                                      cores=config.cores,
                                      disk=genotype_gvcf_disk,
                                      memory=config.xmx)

    # Determine if output GVCF has multiple samples
    if len(gvcfs) == 1:
        uuid = gvcfs.keys()[0]
    else:
        uuid = 'joint_genotyped'

    genotyped_filename = '%s.genotyped%s.vcf' % (uuid, config.suffix)
    genotype_gvcf.addChildJobFn(output_file_job,
                                genotyped_filename,
                                genotype_gvcf.rv(),
                                os.path.join(config.output_dir, uuid),
                                s3_key_path=config.ssec,
                                disk=PromisedRequirement(lambda x: x.size, genotype_gvcf.rv()))

    if config.run_vqsr:
        if not config.joint_genotype:
            job.fileStore.logToMaster('WARNING: Running VQSR without joint genotyping.')
        joint_genotype_vcf = genotype_gvcf.addFollowOnJobFn(vqsr_pipeline,
                                                            uuid,
                                                            genotype_gvcf.rv(),
                                                            config)
    else:
        joint_genotype_vcf = genotype_gvcf.addFollowOnJobFn(hard_filter_pipeline,
                                                            uuid,
                                                            genotype_gvcf.rv(),
                                                            config)
    return joint_genotype_vcf.rv()