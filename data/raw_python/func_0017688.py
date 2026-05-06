def add_read_groups(job, job_vars):
    """
    This function adds read groups to the headers

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    sudo = input_args['sudo']
    # I/O
    alignments = return_input_paths(job, work_dir, ids, 'alignments.bam')
    output = os.path.join(work_dir, 'rg_alignments.bam')
    # Command and callg
    parameter = ['AddOrReplaceReadGroups',
                 'INPUT={}'.format(docker_path(alignments)),
                 'OUTPUT={}'.format(docker_path(output)),
                 'RGSM={}'.format(input_args['uuid']),
                 'RGID={}'.format(input_args['uuid']),
                 'RGLB=TruSeq',
                 'RGPL=illumina',
                 'RGPU=barcode',
                 'VALIDATION_STRINGENCY=SILENT']
    docker_call(tool='quay.io/ucsc_cgl/picardtools:1.95--dd5ac549b95eb3e5d166a5e310417ef13651994e',
                tool_parameters=parameter, work_dir=work_dir, sudo=sudo)
    # Write to FileStore
    ids['rg_alignments.bam'] = job.fileStore.writeGlobalFile(output)
    # Run child job
    return job.addChildJobFn(bamsort_and_index, job_vars, disk='30 G').rv()