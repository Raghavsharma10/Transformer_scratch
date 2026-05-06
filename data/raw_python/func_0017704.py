def upload_data(job, master_ip, inputs, hdfs_name, upload_name, spark_on_toil):
    """
    Upload file hdfsName from hdfs to s3
    """

    if mock_mode():
        truncate_file(master_ip, hdfs_name, spark_on_toil)

    log.info("Uploading output BAM %s to %s.", hdfs_name, upload_name)
    call_conductor(job, master_ip, hdfs_name, upload_name, memory=inputs.memory)
    remove_file(master_ip, hdfs_name, spark_on_toil)