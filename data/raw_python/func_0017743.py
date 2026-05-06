def download_sample_and_align(job, sample, inputs, ids):
    """
    Downloads the sample and runs BWA-kit

    :param JobFunctionWrappingJob job: Passed by Toil automatically
    :param tuple(str, list) sample: UUID and URLS for sample
    :param Namespace inputs: Contains input arguments
    :param dict ids: FileStore IDs for shared inputs
    """
    uuid, urls = sample
    r1_url, r2_url = urls if len(urls) == 2 else (urls[0], None)
    job.fileStore.logToMaster('Downloaded sample: {0}. R1 {1}\nR2 {2}\nStarting BWA Run'.format(uuid, r1_url, r2_url))
    # Read fastq samples from file store
    ids['r1'] = job.addChildJobFn(download_url_job, r1_url, s3_key_path=inputs.ssec, disk=inputs.file_size).rv()
    if r2_url:
        ids['r2'] = job.addChildJobFn(download_url_job, r2_url, s3_key_path=inputs.ssec, disk=inputs.file_size).rv()
    else:
        ids['r2'] = None
    # Create config for bwakit
    inputs.cores = min(inputs.maxCores, multiprocessing.cpu_count())
    inputs.uuid = uuid
    config = dict(**vars(inputs))  # Create config as a copy of inputs since it has values we want
    config.update(ids)  # Overwrite attributes with the FileStoreIDs from ids
    config = argparse.Namespace(**config)
    # Define and wire job functions
    bam_id = job.wrapJobFn(run_bwakit, config, sort=inputs.sort, trim=inputs.trim,
                           disk=inputs.file_size, cores=inputs.cores)
    job.addFollowOn(bam_id)
    output_name = uuid + '.bam' + str(inputs.suffix) if inputs.suffix else uuid + '.bam'
    if urlparse(inputs.output_dir).scheme == 's3':
        bam_id.addChildJobFn(s3am_upload_job, file_id=bam_id.rv(), file_name=output_name, s3_dir=inputs.output_dir,
                             s3_key_path=inputs.ssec, cores=inputs.cores, disk=inputs.file_size)
    else:
        mkdir_p(inputs.ouput_dir)
        bam_id.addChildJobFn(copy_file_job, name=output_name, file_id=bam_id.rv(), output_dir=inputs.output_dir,
                                    disk=inputs.file_size)