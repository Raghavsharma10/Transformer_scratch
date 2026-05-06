def download_run_and_upload(job, master_ip, inputs, spark_on_toil):
    """
    Monolithic job that calls data download, conversion, transform, upload.
    Previously, this was not monolithic; change came in due to #126/#134.
    """
    master_ip = MasterAddress(master_ip)

    bam_name = inputs.sample.split('://')[-1].split('/')[-1]
    sample_name = ".".join(os.path.splitext(bam_name)[:-1])

    hdfs_subdir = sample_name + "-dir"

    if inputs.run_local:
        inputs.local_dir = job.fileStore.getLocalTempDir()
        if inputs.native_adam_path is None:
            hdfs_dir = "/data/"
        else:
            hdfs_dir = inputs.local_dir
    else:
        inputs.local_dir = None
        hdfs_dir = "hdfs://{0}:{1}/{2}".format(master_ip, HDFS_MASTER_PORT, hdfs_subdir)

    try:
        hdfs_prefix = hdfs_dir + "/" + sample_name
        hdfs_bam = hdfs_dir + "/" + bam_name

        hdfs_snps = hdfs_dir + "/" + inputs.dbsnp.split('://')[-1].split('/')[-1]

        if not inputs.run_local:
            download_data(job, master_ip, inputs, inputs.dbsnp, inputs.sample, hdfs_snps, hdfs_bam)
        else:
            copy_files([inputs.sample, inputs.dbsnp], inputs.local_dir)

        adam_input = hdfs_prefix + ".adam"
        adam_snps = hdfs_dir + "/snps.var.adam"
        adam_convert(job, master_ip, inputs, hdfs_bam, hdfs_snps, adam_input, adam_snps, spark_on_toil)

        adam_output = hdfs_prefix + ".processed.bam"
        adam_transform(job, master_ip, inputs, adam_input, adam_snps, hdfs_dir, adam_output, spark_on_toil)

        out_file = inputs.output_dir + "/" + sample_name + inputs.suffix + ".bam"

        if not inputs.run_local:
            upload_data(job, master_ip, inputs, adam_output, out_file, spark_on_toil)
        else:
            local_adam_output = "%s/%s.processed.bam" % (inputs.local_dir, sample_name)
            move_files([local_adam_output], inputs.output_dir)

        remove_file(master_ip, hdfs_subdir, spark_on_toil)
    except:
        remove_file(master_ip, hdfs_subdir, spark_on_toil)
        raise