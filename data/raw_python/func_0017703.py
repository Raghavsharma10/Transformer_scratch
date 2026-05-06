def adam_transform(job, master_ip, inputs, in_file, snp_file, hdfs_dir, out_file, spark_on_toil):
    """
    Preprocess in_file with known SNPs snp_file:
        - mark duplicates
        - realign indels
        - recalibrate base quality scores
    """

    log.info("Marking duplicate reads.")
    call_adam(job, master_ip,
              ["transform",
               in_file,  hdfs_dir + "/mkdups.adam",
               "-aligned_read_predicate",
               "-limit_projection",
               "-mark_duplicate_reads"],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    #FIXME
    in_file_name = in_file.split("/")[-1]
    remove_file(master_ip, in_file_name + "*", spark_on_toil)

    log.info("Realigning INDELs.")
    call_adam(job, master_ip,
              ["transform",
               hdfs_dir + "/mkdups.adam",
               hdfs_dir + "/ri.adam",
               "-realign_indels"],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    remove_file(master_ip, hdfs_dir + "/mkdups.adam*", spark_on_toil)

    log.info("Recalibrating base quality scores.")
    call_adam(job, master_ip,
              ["transform",
               hdfs_dir + "/ri.adam",
               hdfs_dir + "/bqsr.adam",
               "-recalibrate_base_qualities",
               "-known_snps", snp_file],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    remove_file(master_ip, "ri.adam*", spark_on_toil)

    log.info("Sorting reads and saving a single BAM file.")
    call_adam(job, master_ip,
              ["transform",
               hdfs_dir + "/bqsr.adam",
               out_file,
               "-sort_reads", "-single"],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    remove_file(master_ip, "bqsr.adam*", spark_on_toil)

    return out_file