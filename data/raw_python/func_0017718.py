def process_sample(job, inputs, tar_id):
    """
    Converts sample.tar(.gz) into two fastq files.
    Due to edge conditions... BEWARE: HERE BE DRAGONS

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    :param str tar_id: FileStore ID of sample tar
    """
    job.fileStore.logToMaster('Processing sample into read pairs: {}'.format(inputs.uuid))
    work_dir = job.fileStore.getLocalTempDir()
    # I/O
    tar_path = job.fileStore.readGlobalFile(tar_id, os.path.join(work_dir, 'sample.tar'))
    # Untar File and concat
    subprocess.check_call(['tar', '-xvf', tar_path, '-C', work_dir])
    os.remove(os.path.join(work_dir, 'sample.tar'))
    # Grab files from tarball
    fastqs = []
    for root, subdir, files in os.walk(work_dir):
        fastqs.extend([os.path.join(root, x) for x in files])
    # Check for read 1 and read 2 files
    r1 = sorted([x for x in fastqs if 'R1' in x])
    r2 = sorted([x for x in fastqs if 'R2' in x])
    if not r1 or not r2:
        # Check if using a different standard
        r1 = sorted([x for x in fastqs if '_1' in x])
        r2 = sorted([x for x in fastqs if '_2' in x])
    # Prune file name matches from each list
    if len(r1) > len(r2):
        r1 = [x for x in r1 if x not in r2]
    elif len(r2) > len(r1):
        r2 = [x for x in r2 if x not in r1]
    # Flag if data is single-ended
    assert r1 and r2, 'This pipeline does not support single-ended data. R1: {}\nR2:{}'.format(r1, r2)
    command = 'zcat' if r1[0].endswith('gz') and r2[0].endswith('gz') else 'cat'
    with open(os.path.join(work_dir, 'R1.fastq'), 'w') as f1:
        p1 = subprocess.Popen([command] + r1, stdout=f1)
    with open(os.path.join(work_dir, 'R2.fastq'), 'w') as f2:
        p2 = subprocess.Popen([command] + r2, stdout=f2)
    p1.wait()
    p2.wait()
    # Write to fileStore
    r1_id = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R1.fastq'))
    r2_id = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R2.fastq'))
    job.fileStore.deleteGlobalFile(tar_id)
    # Start cutadapt step
    job.addChildJobFn(cutadapt, inputs, r1_id, r2_id, disk='60G').rv()