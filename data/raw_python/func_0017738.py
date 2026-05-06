def static_workflow_declaration(job, config, normal_bam, normal_bai, tumor_bam, tumor_bai):
    """
    Statically declare workflow so sections can be modularly repurposed

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    :param str normal_bam: Normal BAM FileStoreID
    :param str normal_bai: Normal BAM index FileStoreID
    :param str tumor_bam: Tumor BAM FileStoreID
    :param str tumor_bai: Tumor BAM Index FileStoreID
    """
    # Mutation and indel tool wiring
    memory = '1G' if config.ci_test else '10G'
    disk = '1G' if config.ci_test else '75G'
    mutect_results, pindel_results, muse_results = None, None, None
    if config.run_mutect:
        mutect_results = job.addChildJobFn(run_mutect, normal_bam, normal_bai, tumor_bam, tumor_bai, config.reference,
                                           config.dict, config.fai, config.cosmic, config.dbsnp,
                                           cores=1, memory=memory, disk=disk).rv()
    if config.run_pindel:
        pindel_results = job.addChildJobFn(run_pindel, normal_bam, normal_bai, tumor_bam, tumor_bai,
                                           config.reference, config.fai,
                                           cores=config.cores,  memory=memory, disk=disk).rv()
    if config.run_muse:
        muse_results = job.addChildJobFn(run_muse, normal_bam, normal_bai, tumor_bam, tumor_bai,
                                         config.reference, config.dict, config.fai, config.dbsnp,
                                         cores=config.cores, memory=memory, disk=disk).rv()
    # Pass tool results (whether None or a promised return value) to consolidation step
    consolidation = job.wrapJobFn(consolidate_output, config, mutect_results, pindel_results, muse_results)
    job.addFollowOn(consolidation)