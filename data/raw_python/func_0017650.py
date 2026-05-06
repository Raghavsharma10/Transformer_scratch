def annotate_vcfs(job, vcfs, config):
    """
    Runs Oncotator for a group of VCF files. Each sample is annotated individually.

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param dict vcfs: Dictionary of VCF FileStoreIDs {Sample identifier: FileStoreID}
    :param Namespace config: Input parameters and shared FileStoreIDs
        Requires the following config attributes:
        config.oncotator_db         FileStoreID to Oncotator database
        config.suffix               Suffix added to output filename
        config.output_dir           URL or local path to output directory
        config.ssec                 Path to key file for SSE-C encryption
        config.cores                Number of cores for each job
        config.xmx                  Java heap size in bytes
    """
    job.fileStore.logToMaster('Running Oncotator on the following samples:\n%s' % '\n'.join(vcfs.keys()))
    for uuid, vcf_id in vcfs.iteritems():
        # The Oncotator disk requirement depends on the input VCF, the Oncotator database
        # and the output VCF. The annotated VCF will be significantly larger than the input VCF.
        onco_disk = PromisedRequirement(lambda vcf, db: 3 * vcf.size + db.size,
                                        vcf_id,
                                        config.oncotator_db)

        annotated_vcf = job.addChildJobFn(run_oncotator,
                                          vcf_id,
                                          config.oncotator_db,
                                          disk=onco_disk,
                                          cores=config.cores,
                                          memory=config.xmx)

        output_dir = os.path.join(config.output_dir, uuid)
        filename = '{}.oncotator{}.vcf'.format(uuid, config.suffix)
        annotated_vcf.addChildJobFn(output_file_job,
                                    filename,
                                    annotated_vcf.rv(),
                                    output_dir,
                                    s3_key_path=config.ssec,
                                    disk=PromisedRequirement(lambda x: x.size, annotated_vcf.rv()))