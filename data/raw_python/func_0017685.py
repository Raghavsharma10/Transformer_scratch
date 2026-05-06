def static_dag_launchpoint(job, job_vars):
    """
    Statically define jobs in the pipeline

    job_vars: tuple     Tuple of dictionaries: input_args and ids
    """
    input_args, ids = job_vars
    if input_args['config_fastq']:
        cores = input_args['cpu_count']
        a = job.wrapJobFn(mapsplice, job_vars, cores=cores, disk='130G').encapsulate()
    else:
        a = job.wrapJobFn(merge_fastqs, job_vars, disk='70 G').encapsulate()
    b = job.wrapJobFn(consolidate_output, job_vars, a.rv())
    # Take advantage of "encapsulate" to simplify pipeline wiring
    job.addChild(a)
    a.addChild(b)