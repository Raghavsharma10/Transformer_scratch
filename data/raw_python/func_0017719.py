def cutadapt(job, inputs, r1_id, r2_id):
    """
    Filters out adapters that may be left in the RNA-seq files

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    :param str r1_id: FileStore ID of read 1 fastq
    :param str r2_id: FileStore ID of read 2 fastq
    """
    job.fileStore.logToMaster('Running CutAdapt: {}'.format(inputs.uuid))
    work_dir = job.fileStore.getLocalTempDir()
    inputs.improper_pair = None
    # Retrieve files
    job.fileStore.readGlobalFile(r1_id, os.path.join(work_dir, 'R1.fastq'))
    job.fileStore.readGlobalFile(r2_id, os.path.join(work_dir, 'R2.fastq'))
    # Cutadapt parameters
    parameters = ['-a', inputs.fwd_3pr_adapter,
                  '-m', '35',
                  '-A', inputs.rev_3pr_adapter,
                  '-o', '/data/R1_cutadapt.fastq',
                  '-p', '/data/R2_cutadapt.fastq',
                  '/data/R1.fastq', '/data/R2.fastq']
    # Call: CutAdapt
    base_docker_call = 'docker run --log-driver=none --rm -v {}:/data'.format(work_dir).split()
    if inputs.sudo:
        base_docker_call = ['sudo'] + base_docker_call
    tool = 'quay.io/ucsc_cgl/cutadapt:1.9--6bd44edd2b8f8f17e25c5a268fedaab65fa851d2'
    p = subprocess.Popen(base_docker_call + [tool] + parameters, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        if 'improperly paired' in stderr:
            inputs.improper_pair = True
            shutil.move(os.path.join(work_dir, 'R1.fastq'), os.path.join(work_dir, 'R1_cutadapt.fastq'))
            shutil.move(os.path.join(work_dir, 'R2.fastq'), os.path.join(work_dir, 'R2_cutadapt.fastq'))
    # Write to fileStore
    if inputs.improper_pair:
        r1_cutadapt = r1_id
        r2_cutadapt = r2_id
    else:
        r1_cutadapt = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R1_cutadapt.fastq'))
        r2_cutadapt = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'R2_cutadapt.fastq'))
        job.fileStore.deleteGlobalFile(r1_id)
        job.fileStore.deleteGlobalFile(r2_id)
    # start STAR
    cores = min(inputs.cores, 16)
    job.addChildJobFn(star, inputs, r1_cutadapt, r2_cutadapt, cores=cores, disk='100G', memory='40G').rv()