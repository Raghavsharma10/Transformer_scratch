def preprocessing_declaration(job, config):
    """
    Declare jobs related to preprocessing

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    """
    if config.preprocessing:
        job.fileStore.logToMaster('Ran preprocessing: ' + config.uuid)
        disk = '1G' if config.ci_test else '20G'
        mem = '2G' if config.ci_test else '10G'
        processed_normal = job.wrapJobFn(run_gatk_preprocessing, config.normal_bam, config.normal_bai,
                                         config.reference, config.dict, config.fai, config.phase, config.mills,
                                         config.dbsnp, mem, cores=1, memory=mem, disk=disk)
        processed_tumor = job.wrapJobFn(run_gatk_preprocessing, config.tumor_bam, config.tumor_bai,
                                        config.reference, config.dict, config.fai, config.phase, config.mills,
                                        config.dbsnp, mem, cores=1, memory=mem, disk=disk)
        static_workflow = job.wrapJobFn(static_workflow_declaration, config, processed_normal.rv(0),
                                        processed_normal.rv(1), processed_tumor.rv(0), processed_tumor.rv(1))
        job.addChild(processed_normal)
        job.addChild(processed_tumor)
        job.addFollowOn(static_workflow)
    else:
        job.addFollowOnJobFn(static_workflow_declaration, config, config.normal_bam, config.normal_bai,
                             config.tumor_bam, config.tumor_bai)