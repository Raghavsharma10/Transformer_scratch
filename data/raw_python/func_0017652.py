def download_shared_files(job, config):
    """
    Downloads shared reference files for Toil Germline pipeline

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Pipeline configuration options
    :return: Updated config with shared fileStoreIDS
    :rtype: Namespace
    """
    job.fileStore.logToMaster('Downloading shared reference files')
    shared_files = {'genome_fasta', 'genome_fai', 'genome_dict'}
    nonessential_files = {'genome_fai', 'genome_dict'}

    # Download necessary files for pipeline configuration
    if config.run_bwa:
        shared_files |= {'amb', 'ann', 'bwt', 'pac', 'sa', 'alt'}
        nonessential_files.add('alt')
    if config.preprocess:
        shared_files |= {'g1k_indel', 'mills', 'dbsnp'}
    if config.run_vqsr:
        shared_files |= {'g1k_snp', 'mills', 'dbsnp', 'hapmap', 'omni'}
    if config.run_oncotator:
        shared_files.add('oncotator_db')
    for name in shared_files:
        try:
            url = getattr(config, name, None)
            if url is None:
                continue
            setattr(config, name, job.addChildJobFn(download_url_job,
                                                    url,
                                                    name=name,
                                                    s3_key_path=config.ssec,
                                                    disk='15G'   # Estimated reference file size
                                                    ).rv())
        finally:
            if getattr(config, name, None) is None and name not in nonessential_files:
                raise ValueError("Necessary configuration parameter is missing:\n{}".format(name))
    return job.addFollowOnJobFn(reference_preprocessing, config).rv()