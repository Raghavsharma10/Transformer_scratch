def rseq_qc(job, job_vars):
    """
    QC module: contains QC metrics and information about the BAM post alignment

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    uuid = input_args['uuid']
    sudo = input_args['sudo']
    # I/O
    return_input_paths(job, work_dir, ids, 'sorted.bam', 'sorted.bam.bai')
    # Command
    docker_call(tool='jvivian/qc', tool_parameters=['/opt/cgl-docker-lib/RseqQC_v2.sh', '/data/sorted.bam', uuid],
                work_dir=work_dir, sudo=sudo)
    # Write to FileStore
    output_files = [f for f in glob.glob(os.path.join(work_dir, '*')) if 'sorted.bam' not in f]
    tarball_files(work_dir, tar_name='qc.tar.gz', uuid=None, files=output_files)
    return job.fileStore.writeGlobalFile(os.path.join(work_dir, 'qc.tar.gz'))