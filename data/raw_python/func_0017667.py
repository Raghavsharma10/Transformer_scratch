def alignment(job, ids, input_args, sample):
    """
    Runs BWA and then Bamsort on the supplied fastqs for this sample

    Input1: Toil Job instance
    Input2: jobstore id dictionary
    Input3: Input arguments dictionary
    Input4: Sample tuple -- contains uuid and urls for the sample
    """
    uuid, urls = sample
    # ids['bam'] = job.fileStore.getEmptyFileStoreID()
    work_dir = job.fileStore.getLocalTempDir()
    output_dir = input_args['output_dir']
    key_path = input_args['ssec']
    cores = multiprocessing.cpu_count()

    # I/O
    return_input_paths(job, work_dir, ids, 'ref.fa', 'ref.fa.amb', 'ref.fa.ann',
                                                     'ref.fa.bwt', 'ref.fa.pac', 'ref.fa.sa', 'ref.fa.fai')
    # Get fastqs associated with this sample
    for url in urls:
        download_encrypted_file(work_dir, url, key_path, os.path.basename(url))

    # Parameters for BWA and Bamsort
    docker_cmd = ['docker', 'run', '--rm', '-v', '{}:/data'.format(work_dir)]

    bwa_command = ["jvivian/bwa",
                   "mem",
                   "-R", "@RG\tID:{0}\tPL:Illumina\tSM:{0}\tLB:KapaHyper".format(uuid),
                   "-T", str(0),
                   "-t", str(cores),
                   "/data/ref.fa"] + [os.path.join('/data/',  os.path.basename(x)) for x in urls]

    bamsort_command = ["jeltje/biobambam",
                       "/usr/local/bin/bamsort",
                       "inputformat=sam",
                       "level=1",
                       "inputthreads={}".format(cores),
                       "outputthreads={}".format(cores),
                       "calmdnm=1",
                       "calmdnmrecompindetonly=1",
                       "calmdnmreference=/data/ref.fa",
                       "I=/data/{}".format(uuid + '.sam')]
    # Piping the output to a file handle
    with open(os.path.join(work_dir, uuid + '.sam'), 'w') as f_out:
        subprocess.check_call(docker_cmd + bwa_command, stdout=f_out)

    with open(os.path.join(work_dir, uuid + '.bam'), 'w') as f_out:
        subprocess.check_call(docker_cmd + bamsort_command, stdout=f_out)

    # Save in JobStore
    # job.fileStore.updateGlobalFile(ids['bam'], os.path.join(work_dir, uuid + '.bam'))
    ids['bam'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, uuid + '.bam'))
    # Copy file to S3
    if input_args['s3_dir']:
        job.addChildJobFn(upload_bam_to_s3, ids, input_args, sample, cores=32, memory='20 G', disk='30 G')
    # Move file in output_dir
    if input_args['output_dir']:
        move_to_output_dir(work_dir, output_dir, uuid=None, files=[uuid + '.bam'])