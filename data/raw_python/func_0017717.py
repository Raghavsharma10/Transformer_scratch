def download_sample(job, sample, inputs):
    """
    Download the input sample

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param tuple sample: Tuple containing (UUID,URL) of a sample
    :param Namespace inputs: Stores input arguments (see main)
    """
    uuid, url = sample
    job.fileStore.logToMaster('Downloading sample: {}'.format(uuid))
    # Download sample
    tar_id = job.addChildJobFn(download_url_job, url, s3_key_path=inputs.ssec, disk='30G').rv()
    # Create copy of inputs for each sample
    sample_inputs = argparse.Namespace(**vars(inputs))
    sample_inputs.uuid = uuid
    sample_inputs.cores = multiprocessing.cpu_count()
    # Call children and follow-on jobs
    job.addFollowOnJobFn(process_sample, sample_inputs, tar_id, cores=2, disk='60G')