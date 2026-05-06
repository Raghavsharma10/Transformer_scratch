def merge_fastqs(job, job_vars):
    """
    Unzips input sample and concats the Read1 and Read2 groups together.

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    cores = input_args['cpu_count']
    single_end_reads = input_args['single_end_reads']
    # I/O
    sample = return_input_paths(job, work_dir, ids, 'sample.tar')
    # Untar File
    # subprocess.check_call(['unzip', sample, '-d', work_dir])
    subprocess.check_call(['tar', '-xvf', sample, '-C', work_dir])
    # Remove large files before creating concat versions.
    os.remove(os.path.join(work_dir, 'sample.tar'))
    # Zcat files in parallel
    if single_end_reads:
        files = sorted(glob.glob(os.path.join(work_dir, '*')))
        with open(os.path.join(work_dir, 'R1.fastq'), 'w') as f1:
            subprocess.check_call(['zcat'] + files, stdout=f1)
        # FileStore
        ids['R1.fastq'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R1.fastq'))
    else:
        r1_files = sorted(glob.glob(os.path.join(work_dir, '*R1*')))
        r2_files = sorted(glob.glob(os.path.join(work_dir, '*R2*')))
        with open(os.path.join(work_dir, 'R1.fastq'), 'w') as f1:
            p1 = subprocess.Popen(['zcat'] + r1_files, stdout=f1)
        with open(os.path.join(work_dir, 'R2.fastq'), 'w') as f2:
            p2 = subprocess.Popen(['zcat'] + r2_files, stdout=f2)
        p1.wait()
        p2.wait()
        # FileStore
        ids['R1.fastq'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R1.fastq'))
        ids['R2.fastq'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R2.fastq'))
    job.fileStore.deleteGlobalFile(ids['sample.tar'])
    # Spawn child job
    return job.addChildJobFn(mapsplice, job_vars, cores=cores, disk='130 G').rv()