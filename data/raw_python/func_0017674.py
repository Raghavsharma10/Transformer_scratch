def download_and_transfer_sample(job, input_args, samples):
    """
    Downloads a sample from dbGaP via SRAToolKit, then uses S3AM to transfer it to S3

    input_args: dict        Dictionary of input arguments
    analysis_id: str        An analysis ID for a sample in CGHub
    """
    if len(samples) > 1:
        a = samples[len(samples)/2:]
        b = samples[:len(samples)/2]
        job.addChildJobFn(download_and_transfer_sample, input_args, a, disk='30G')
        job.addChildJobFn(download_and_transfer_sample, input_args, b, disk='30G')
    else:
        analysis_id = samples[0]
        work_dir = job.fileStore.getLocalTempDir()
        sudo = input_args['sudo']
        # Acquire dbgap_key
        shutil.copy(input_args['dbgap_key'], os.path.join(work_dir, 'dbgap.ngc'))
        # Call to fastq-dump to pull down SRA files and convert to fastq
        if input_args['single_end']:
            parameters = [analysis_id]
        else:
            parameters = ['--split-files', analysis_id]
        docker_call(tool='quay.io/ucsc_cgl/fastq-dump:2.5.7--4577a6c1a3c94adaa0c25dd6c03518ee610433d1',
                    work_dir=work_dir, tool_parameters=parameters, sudo=sudo)
        # Collect files and encapsulate into a tarball
        shutil.rmtree(os.path.join(work_dir, 'sra'))
        sample_name = analysis_id + '.tar.gz'
        if input_args['single_end']:
            r = [os.path.basename(x) for x in glob.glob(os.path.join(work_dir, '*.f*'))]
            tarball_files(work_dir, tar_name=sample_name, files=r)
        else:
            r1 = [os.path.basename(x) for x in glob.glob(os.path.join(work_dir, '*_1*'))]
            r2 = [os.path.basename(x) for x in glob.glob(os.path.join(work_dir, '*_2*'))]
            tarball_files(work_dir, tar_name=sample_name, files=r1 + r2)
        # Parse s3_dir to get bucket and s3 path
        key_path = input_args['ssec']
        s3_dir = input_args['s3_dir']
        bucket_name = s3_dir.lstrip('/').split('/')[0]
        base_url = 'https://s3-us-west-2.amazonaws.com/'
        url = os.path.join(base_url, bucket_name, sample_name)
        # Generate keyfile for upload
        with open(os.path.join(work_dir, 'temp.key'), 'wb') as f_out:
            f_out.write(generate_unique_key(key_path, url))
        # Upload to S3 via S3AM
        s3am_command = ['s3am',
                        'upload',
                        '--sse-key-file', os.path.join(work_dir, 'temp.key'),
                        'file://{}'.format(os.path.join(work_dir, sample_name)),
                        's3://' + bucket_name + '/']
        subprocess.check_call(s3am_command)