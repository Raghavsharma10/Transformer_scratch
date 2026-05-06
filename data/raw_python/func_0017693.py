def transcriptome(job, job_vars):
    """
    Creates a bam of just the transcriptome

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    sudo = input_args['sudo']
    # I/O
    sort_by_ref, bed, hg19_fa = return_input_paths(job, work_dir, ids, 'sort_by_ref.bam',
                                                   'unc.bed', 'hg19.transcripts.fa')
    output = os.path.join(work_dir, 'transcriptome.bam')
    # Command
    parameters = ['sam-xlate',
                  '--bed', docker_path(bed),
                  '--in', docker_path(sort_by_ref),
                  '--order', docker_path(hg19_fa),
                  '--out', docker_path(output),
                  '--xgtag',
                  '--reverse']
    docker_call(tool='quay.io/ucsc_cgl/ubu:1.2--02806964cdf74bf5c39411b236b4c4e36d026843',
                tool_parameters=parameters, work_dir=work_dir, java_opts='-Xmx30g', sudo=sudo)
    # Write to FileStore
    ids['transcriptome.bam'] = job.fileStore.writeGlobalFile(output)
    # Run child job
    return job.addChildJobFn(filter_bam, job_vars, memory='30G', disk='30G').rv()