def upload_output_to_s3(job, job_vars):
    """
    If s3_dir is specified in arguments, file will be uploaded to S3 using boto.
    WARNING: ~/.boto credentials are necessary for this to succeed!

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    import boto
    from boto.s3.key import Key

    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    uuid = input_args['uuid']
    # Parse s3_dir
    s3_dir = input_args['s3_dir']
    bucket_name = s3_dir.split('/')[0]
    bucket_dir = '/'.join(s3_dir.split('/')[1:])
    # I/O
    uuid_tar = return_input_paths(job, work_dir, ids, 'uuid.tar.gz')
    # Upload to S3 via boto
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    k = Key(bucket)
    k.key = os.path.join(bucket_dir, uuid + '.tar.gz')
    k.set_contents_from_filename(uuid_tar)