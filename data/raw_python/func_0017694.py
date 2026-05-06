def filter_bam(job, job_vars):
    """
    Performs filtering on the transcriptome bam

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    cores = input_args['cpu_count']
    sudo = input_args['sudo']
    # I/O
    transcriptome_bam = return_input_paths(job, work_dir, ids, 'transcriptome.bam')
    output = os.path.join(work_dir, 'filtered.bam')
    # Command
    parameters = ['sam-filter',
                  '--strip-indels',
                  '--max-insert', '1000',
                  '--mapq', '1',
                  '--in', docker_path(transcriptome_bam),
                  '--out', docker_path(output)]
    docker_call(tool='quay.io/ucsc_cgl/ubu:1.2--02806964cdf74bf5c39411b236b4c4e36d026843',
                tool_parameters=parameters, work_dir=os.path.dirname(output), java_opts='-Xmx30g', sudo=sudo)
    # Write to FileStore
    ids['filtered.bam'] = job.fileStore.writeGlobalFile(output)
    # Run child job
    return job.addChildJobFn(rsem, job_vars, cores=cores, disk='30 G').rv()