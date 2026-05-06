def static_adam_preprocessing_dag(job, inputs, sample, output_dir, suffix=''):
    """
    A Toil job function performing ADAM preprocessing on a single sample
    """
    inputs.sample = sample
    inputs.output_dir = output_dir
    inputs.suffix = suffix

    if inputs.master_ip is not None or inputs.run_local:
        if not inputs.run_local and inputs.master_ip == 'auto':
            # Static, standalone Spark cluster managed by uberscript
            spark_on_toil = False
            scale_up = job.wrapJobFn(scale_external_spark_cluster, 1)
            job.addChild(scale_up)
            spark_work = job.wrapJobFn(download_run_and_upload,
                                       inputs.master_ip, inputs, spark_on_toil)
            scale_up.addChild(spark_work)
            scale_down = job.wrapJobFn(scale_external_spark_cluster, -1)
            spark_work.addChild(scale_down)
        else:
            # Static, external Spark cluster
            spark_on_toil = False
            spark_work = job.wrapJobFn(download_run_and_upload,
                                       inputs.master_ip, inputs, spark_on_toil)
            job.addChild(spark_work)
    else:
        # Dynamic subclusters, i.e. Spark-on-Toil
        spark_on_toil = True
        cores = multiprocessing.cpu_count()
        master_ip = spawn_spark_cluster(job,
                                        False, # Sudo
                                        inputs.num_nodes-1,
                                        cores=cores,
                                        memory=inputs.memory)
        spark_work = job.wrapJobFn(download_run_and_upload,
                                   master_ip, inputs, spark_on_toil)
        job.addChild(spark_work)