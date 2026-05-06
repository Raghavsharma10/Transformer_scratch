def reference_preprocessing(job, samples, config):
    """
    Spawn the jobs that create index and dict file for reference

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param Namespace config: Argparse Namespace object containing argument inputs
    :param list[list] samples: A nested list of samples containing sample information
    """
    job.fileStore.logToMaster('Processed reference files')
    config.fai = job.addChildJobFn(run_samtools_faidx, config.reference).rv()
    config.dict = job.addChildJobFn(run_picard_create_sequence_dictionary, config.reference).rv()
    job.addFollowOnJobFn(map_job, download_sample, samples, config)