def download_sample(job, sample, config):
    """
    Download sample and store sample specific attributes

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param list sample: Contains uuid, normal URL, and tumor URL
    :param Namespace config: Argparse Namespace object containing argument inputs
    """
    # Create copy of config that is sample specific
    config = argparse.Namespace(**vars(config))
    uuid, normal_url, tumor_url = sample
    job.fileStore.logToMaster('Downloaded sample: ' + uuid)
    config.uuid = uuid
    config.normal = normal_url
    config.tumor = tumor_url
    config.cores = min(config.maxCores, int(multiprocessing.cpu_count()))
    disk = '1G' if config.ci_test else '20G'
    # Download sample bams and launch pipeline
    config.normal_bam = job.addChildJobFn(download_url_job, url=config.normal, s3_key_path=config.ssec,
                                          cghub_key_path=config.gtkey, disk=disk).rv()
    config.tumor_bam = job.addChildJobFn(download_url_job, url=config.tumor, s3_key_path=config.ssec,
                                         cghub_key_path=config.gtkey, disk=disk).rv()
    job.addFollowOnJobFn(index_bams, config)