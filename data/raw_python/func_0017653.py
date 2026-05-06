def reference_preprocessing(job, config):
    """
    Creates a genome fasta index and sequence dictionary file if not already present in the pipeline config.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Pipeline configuration options and shared files.
                             Requires FileStoreID for genome fasta file as config.genome_fasta
    :return: Updated config with reference index files
    :rtype: Namespace
    """
    job.fileStore.logToMaster('Preparing Reference Files')
    genome_id = config.genome_fasta
    if getattr(config, 'genome_fai', None) is None:
        config.genome_fai = job.addChildJobFn(run_samtools_faidx,
                                              genome_id,
                                              cores=config.cores).rv()
    if getattr(config, 'genome_dict', None) is None:
        config.genome_dict = job.addChildJobFn(run_picard_create_sequence_dictionary,
                                               genome_id,
                                               cores=config.cores,
                                               memory=config.xmx).rv()
    return config