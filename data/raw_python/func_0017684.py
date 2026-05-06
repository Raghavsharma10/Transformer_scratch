def download_sample(job, ids, input_args, sample):
    """
    Defines variables unique to a sample that are used in the rest of the pipelines

    ids: dict           Dictionary of fileStore IDS
    input_args: dict    Dictionary of input arguments
    sample: tuple       Contains uuid and sample_url
    """
    if len(sample) == 2:
        uuid, sample_location = sample
        url1, url2 = None, None
    else:
        uuid, url1, url2 = sample
        sample_location = None
    # Update values unique to sample
    sample_input = dict(input_args)
    sample_input['uuid'] = uuid
    sample_input['sample.tar'] = sample_location
    if sample_input['output_dir']:
        sample_input['output_dir'] = os.path.join(input_args['output_dir'], uuid)
    sample_input['cpu_count'] = multiprocessing.cpu_count()
    job_vars = (sample_input, ids)
    # Download or locate local file and place in the jobStore
    if sample_input['input']:
        ids['sample.tar'] = job.fileStore.writeGlobalFile(os.path.abspath(sample_location))
    elif sample_input['config_fastq']:
        ids['R1.fastq'] = job.fileStore.writeGlobalFile(urlparse(url1).path)
        ids['R2.fastq'] = job.fileStore.writeGlobalFile(urlparse(url2).path)
    else:
        if sample_input['ssec']:
            ids['sample.tar'] = job.addChildJobFn(download_encrypted_file, sample_input, 'sample.tar', disk='25G').rv()
        else:
            ids['sample.tar'] = job.addChildJobFn(download_from_url, sample_input['sample.tar'], disk='25G').rv()
    job.addFollowOnJobFn(static_dag_launchpoint, job_vars)