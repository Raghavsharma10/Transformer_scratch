def bamsort_and_index(job, job_vars):
    """
    Sorts bam file and produces index file

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    # Unpack variables
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    sudo = input_args['sudo']
    # I/O
    rg_alignments = return_input_paths(job, work_dir, ids, 'rg_alignments.bam')
    output = os.path.join(work_dir, 'sorted.bam')
    # Command -- second argument is "Output Prefix"
    cmd1 = ['sort', docker_path(rg_alignments), docker_path('sorted')]
    cmd2 = ['index', docker_path(output)]
    docker_call(tool='quay.io/ucsc_cgl/samtools:0.1.19--dd5ac549b95eb3e5d166a5e310417ef13651994e',
                tool_parameters=cmd1, work_dir=work_dir, sudo=sudo)
    docker_call(tool='quay.io/ucsc_cgl/samtools:0.1.19--dd5ac549b95eb3e5d166a5e310417ef13651994e',
                tool_parameters=cmd2, work_dir=work_dir, sudo=sudo)
    # Write to FileStore
    ids['sorted.bam'] = job.fileStore.writeGlobalFile(output)
    ids['sorted.bam.bai'] = job.fileStore.writeGlobalFile(os.path.join(work_dir, 'sorted.bam.bai'))
    # Run child job
    output_ids = job.addChildJobFn(sort_bam_by_reference, job_vars, disk='50 G').rv()
    rseq_id = job.addChildJobFn(rseq_qc, job_vars, disk='20 G').rv()
    return rseq_id, output_ids