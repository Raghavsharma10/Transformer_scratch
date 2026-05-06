def joint_genotype_and_filter(job, gvcfs, config):
    """
    Checks for enough disk space for joint genotyping, then calls the genotype and filter pipeline function.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param dict gvcfs: Dictionary of GVCFs {Sample ID: FileStoreID}
    :param Namespace config: Input parameters and reference FileStoreIDs
        Requires the following config attributes:
        config.genome_fasta         FilesStoreID for reference genome fasta file
        config.genome_fai           FilesStoreID for reference genome fasta index file
        config.genome_dict          FilesStoreID for reference genome sequence dictionary file
        config.available_disk       Total available disk space
    :returns: FileStoreID for the joint genotyped and filtered VCF file
    :rtype: str
    """
    # Get the total size of genome reference files
    genome_ref_size = config.genome_fasta.size + config.genome_fai.size + config.genome_dict.size

    # Require at least 2.5x the sum of the individual GVCF files
    cohort_size = sum(gvcf.size for gvcf in gvcfs.values())
    require(int(2.5 * cohort_size + genome_ref_size) < config.available_disk,
            'There is not enough disk space to joint '
            'genotype samples:\n{}'.format('\n'.join(gvcfs.keys())))

    job.fileStore.logToMaster('Merging cohort into a single GVCF file')

    return job.addChildJobFn(genotype_and_filter, gvcfs, config).rv()