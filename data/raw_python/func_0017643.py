def kmer_dag(job,
             input_file,
             output_path,
             kmer_length,
             spark_conf,
             workers,
             cores,
             memory,
             sudo):
    '''
    Optionally launches a Spark cluster and then runs ADAM to count k-mers on an
    input file.

    :param job: Toil job
    :param input_file: URL/path to input file to count k-mers on
    :param output_path: URL/path to save k-mer counts at
    :param kmer_length: The length of k-mer substrings to count.
    :param spark_conf: Optional Spark configuration. If set, workers should \
    not be set.
    :param workers: Optional number of Spark workers to launch. If set, \
    spark_conf should not be set, and cores and memory should be set.
    :param cores: Number of cores per Spark worker. Must be set if workers is \
    set.
    :param memory: Amount of memory to provided to Spark workers. Must be set \
    if workers is set.
    :param sudo: Whether or not to run Spark containers with sudo.

    :type job: toil.Job
    :type input_file: string
    :type output_path: string
    :type kmer_length: int or string
    :type spark_conf: string or None
    :type workers: int or None
    :type cores: int or None
    :type memory: int or None
    :type sudo: boolean
    '''

    require((spark_conf is not None and workers is None) or
            (workers is not None and cores is not None and memory is not None and spark_conf is not None),
            "Either worker count (--workers) must be defined or user must pass in Spark configuration (--spark-conf).")

    # if we do not have a spark configuration, then we must spawn a cluster
    if spark_conf is None:
        master_hostname = spawn_spark_cluster(job,
                                              sudo,
                                              workers,
                                              cores)
    else:
        spark_conf = shlex.split(spark_conf)

    job.addChildJobFn(download_count_upload,
                      masterHostname,
                      input_file, output_file, kmer_length,
                      spark_conf, memory, sudo)