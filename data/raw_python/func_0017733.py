def download_shared_files(job, samples, config):
    """
    Downloads files shared by all samples in the pipeline

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    :param list[list] samples: A nested list of samples containing sample information
    """
    job.fileStore.logToMaster('Downloaded shared files')
    file_names = ['reference', 'phase', 'mills', 'dbsnp', 'cosmic']
    urls = [config.reference, config.phase, config.mills, config.dbsnp, config.cosmic]
    for name, url in zip(file_names, urls):
        if url:
            vars(config)[name] = job.addChildJobFn(download_url_job, url=url).rv()
    job.addFollowOnJobFn(reference_preprocessing, samples, config)