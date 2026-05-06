def upload_bam_to_s3(job, ids, input_args, sample):
    """
    Uploads output BAM from sample to S3

    Input1: Toil Job instance
    Input2: jobstore id dictionary
    Input3: Input arguments dictionary
    Input4: Sample tuple -- contains uuid and urls for the sample
    """
    uuid, urls = sample
    key_path = input_args['ssec']
    work_dir = job.fileStore.getLocalTempDir()
    # Parse s3_dir to get bucket and s3 path
    s3_dir = input_args['s3_dir']
    bucket_name = s3_dir.lstrip('/').split('/')[0]
    bucket_dir = '/'.join(s3_dir.lstrip('/').split('/')[1:])
    base_url = 'https://s3-us-west-2.amazonaws.com/'
    url = os.path.join(base_url, bucket_name, bucket_dir, uuid + '.bam')
    #I/O
    job.fileStore.readGlobalFile(ids['bam'], os.path.join(work_dir, uuid + '.bam'))
    # Generate keyfile for upload
    with open(os.path.join(work_dir, uuid + '.key'), 'wb') as f_out:
        f_out.write(generate_unique_key(key_path, url))
    # Commands to upload to S3 via S3AM
    s3am_command = ['s3am',
                    'upload',
                    '--sse-key-file', os.path.join(work_dir, uuid + '.key'),
                    'file://{}'.format(os.path.join(work_dir, uuid + '.bam')),
                    bucket_name,
                    os.path.join(bucket_dir, uuid + '.bam')]

    subprocess.check_call(s3am_command)