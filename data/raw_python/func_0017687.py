def mapsplice(job, job_vars):
    """
    Maps RNA-Seq reads to a reference genome.

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    # Unpack variables
    input_args, ids = job_vars
    work_dir = job.fileStore.getLocalTempDir()
    cores = input_args['cpu_count']
    sudo = input_args['sudo']
    single_end_reads = input_args['single_end_reads']
    files_to_delete = ['R1.fastq']
    # I/O
    return_input_paths(job, work_dir, ids, 'ebwt.zip', 'chromosomes.zip')
    if single_end_reads:
        return_input_paths(job, work_dir, ids, 'R1.fastq')
    else:
        return_input_paths(job, work_dir, ids, 'R1.fastq', 'R2.fastq')
        files_to_delete.extend(['R2.fastq'])
    for fname in ['chromosomes.zip', 'ebwt.zip']:
        subprocess.check_call(['unzip', '-o', os.path.join(work_dir, fname), '-d', work_dir])
    # Command and call
    parameters = ['-p', str(cores),
                  '-s', '25',
                  '--bam',
                  '--min-map-len', '50',
                  '-x', '/data/ebwt',
                  '-c', '/data/chromosomes',
                  '-1', '/data/R1.fastq',
                  '-o', '/data']
    if not single_end_reads:
        parameters.extend(['-2', '/data/R2.fastq'])
    docker_call(tool='quay.io/ucsc_cgl/mapsplice:2.1.8--dd5ac549b95eb3e5d166a5e310417ef13651994e',
                tool_parameters=parameters, work_dir=work_dir, sudo=sudo)
    # Write to FileStore
    for fname in ['alignments.bam', 'stats.txt']:
        ids[fname] = job.fileStore.writeGlobalFile(os.path.join(work_dir, fname))
    for fname in files_to_delete:
        job.fileStore.deleteGlobalFile(ids[fname])
    # Run child job
    # map_id = job.addChildJobFn(mapping_stats, job_vars).rv()
    if input_args['upload_bam_to_s3'] and input_args['s3_dir']:
        job.addChildJobFn(upload_bam_to_s3, job_vars)
    output_ids = job.addChildJobFn(add_read_groups, job_vars, disk='30 G').rv()
    return output_ids