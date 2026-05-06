def rsem(job, job_vars):
    """
    Runs RSEM to produce counts

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    cpus = input_args['cpu_count']
    sudo = input_args['sudo']
    single_end_reads = input_args['single_end_reads']
    # I/O
    filtered_bam, rsem_ref = return_input_paths(job, work_dir, ids, 'filtered.bam', 'rsem_ref.zip')
    subprocess.check_call(['unzip', '-o', os.path.join(work_dir, 'rsem_ref.zip'), '-d', work_dir])
    output_prefix = 'rsem'
    # Make tool call to Docker
    parameters = ['--quiet',
                  '--no-qualities',
                  '-p', str(cpus),
                  '--forward-prob', '0.5',
                  '--seed-length', '25',
                  '--fragment-length-mean', '-1.0',
                  '--bam', docker_path(filtered_bam)]
    if not single_end_reads:
        parameters.extend(['--paired-end'])
    parameters.extend(['/data/rsem_ref/hg19_M_rCRS_ref', output_prefix])

    docker_call(tool='quay.io/ucsc_cgl/rsem:1.2.25--4e8d1b31d4028f464b3409c6558fb9dfcad73f88',
                tool_parameters=parameters, work_dir=work_dir, sudo=sudo)
    os.rename(os.path.join(work_dir, output_prefix + '.genes.results'), os.path.join(work_dir, 'rsem_gene.tab'))
    os.rename(os.path.join(work_dir, output_prefix + '.isoforms.results'), os.path.join(work_dir, 'rsem_isoform.tab'))
    # Write to FileStore
    ids['rsem_gene.tab'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'rsem_gene.tab'))
    ids['rsem_isoform.tab'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'rsem_isoform.tab'))
    # Run child jobs
    return job.addChildJobFn(rsem_postprocess, job_vars).rv()