def adam_convert(job, master_ip, inputs, in_file, in_snps, adam_file, adam_snps, spark_on_toil):
    """
    Convert input sam/bam file and known SNPs file into ADAM format
    """

    log.info("Converting input BAM to ADAM.")
    call_adam(job, master_ip,
              ["transform", in_file, adam_file],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    in_file_name = in_file.split("/")[-1]
    remove_file(master_ip, in_file_name, spark_on_toil)

    log.info("Converting known sites VCF to ADAM.")

    call_adam(job, master_ip,
              ["vcf2adam", "-only_variants", in_snps, adam_snps],
              memory=inputs.memory,
              run_local=inputs.run_local,
              native_adam_path=inputs.native_adam_path)

    in_snps_name = in_snps.split("/")[-1]
    remove_file(master_ip, in_snps_name, spark_on_toil)