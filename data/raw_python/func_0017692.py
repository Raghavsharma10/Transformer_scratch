def exon_count(job, job_vars):
    """
    Produces exon counts

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    uuid = input_args['uuid']
    sudo = input_args['sudo']
    # I/O
    sort_by_ref, normalize_pl, composite_bed = return_input_paths(job, work_dir, ids, 'sort_by_ref.bam',
                                                                  'normalize.pl', 'composite_exons.bed')
    # Command
    tool = 'jvivian/bedtools'
    cmd_1 = ['coverage',
             '-split',
             '-abam', docker_path(sort_by_ref),
             '-b', docker_path(composite_bed)]
    cmd_2 = ['perl',
             os.path.join(work_dir, 'normalize.pl'),
             sort_by_ref,
             composite_bed]

    popen_docker = ['docker', 'run', '-v', '{}:/data'.format(work_dir), tool]
    if sudo:
        popen_docker = ['sudo'] + popen_docker
    p = subprocess.Popen(popen_docker + cmd_1, stdout=subprocess.PIPE)
    with open(os.path.join(work_dir, 'exon_quant'), 'w') as f:
        subprocess.check_call(cmd_2, stdin=p.stdout, stdout=f)

    p1 = subprocess.Popen(['cat', os.path.join(work_dir, 'exon_quant')], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['tr', '":"', '"\t"'], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(['tr', '"-"', '"\t"'], stdin=p2.stdout, stdout=subprocess.PIPE)
    with open(os.path.join(work_dir, 'exon_quant.bed'), 'w') as f:
        subprocess.check_call(['cut', '-f1-4'], stdin=p3.stdout, stdout=f)
    # Create zip, upload to fileStore, and move to output_dir as a backup
    output_files = ['exon_quant.bed', 'exon_quant']
    tarball_files(work_dir, tar_name='exon.tar.gz', uuid=uuid, files=output_files)
    return job.fileStore.writeGlobalFile(os.path.join(work_dir, 'exon.tar.gz'))