def download_data(job, master_ip, inputs, known_snps, bam, hdfs_snps, hdfs_bam):
    """
    Downloads input data files from S3.

    :type masterIP: MasterAddress
    """

    log.info("Downloading known sites file %s to %s.", known_snps, hdfs_snps)
    call_conductor(job, master_ip, known_snps, hdfs_snps, memory=inputs.memory)

    log.info("Downloading input BAM %s to %s.", bam, hdfs_bam)
    call_conductor(job, master_ip, bam, hdfs_bam, memory=inputs.memory)