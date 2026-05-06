def index_bams(job, config):
    """
    Convenience job for handling bam indexing to make the workflow declaration cleaner

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    """
    job.fileStore.logToMaster('Indexed sample BAMS: ' + config.uuid)
    disk = '1G' if config.ci_test else '20G'
    config.normal_bai = job.addChildJobFn(run_samtools_index, config.normal_bam, cores=1, disk=disk).rv()
    config.tumor_bai = job.addChildJobFn(run_samtools_index, config.tumor_bam, cores=1, disk=disk).rv()
    job.addFollowOnJobFn(preprocessing_declaration, config)