def upload_bam_to_s3(job, job_vars):
    """
    Upload bam to S3. Requires S3AM and a ~/.boto config file.
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    uuid = input_args['uuid']
    # I/O
    job.fileStore.readGlobalFile(ids['alignments.bam'], os.path.join(work_dir, 'alignments.bam'))
    bam_path = os.path.join(work_dir, 'alignments.bam')
    sample_name = uuid + '.bam'
    # Parse s3_dir to get bucket and s3 path
    s3_dir = input_args['s3_dir']
    bucket_name = s3_dir.split('/')[0]
    bucket_dir = os.path.join('/'.join(s3_dir.split('/')[1:]), 'bam_files')
    # Upload to S3 via S3AM
    s3am_command = ['s3am',
                    'upload',
                    'file://{}'.format(bam_path),
                    os.path.join('s3://', bucket_name, bucket_dir, sample_name)]
    subprocess.check_call(s3am_command)