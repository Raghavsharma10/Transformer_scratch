def run_gatk_germline_pipeline(job, samples, config):
    """
    Downloads shared files and calls the GATK best practices germline pipeline for a cohort of samples

    :param JobFunctionWrappingJob job: passed automatically by Toil
    :param list[GermlineSample] samples: List of GermlineSample namedtuples
    :param Namespace config: Configuration options for pipeline
        Requires the following config attributes:
        config.preprocess_only      If True, then stops pipeline after preprocessing steps
        config.joint_genotype       If True, then joint genotypes cohort
        config.run_oncotator        If True, then adds Oncotator to pipeline
        Additional parameters are needed for downstream steps. Refer to pipeline README for more information.
    """
    # Determine the available disk space on a worker node before any jobs have been run.
    work_dir = job.fileStore.getLocalTempDir()
    st = os.statvfs(work_dir)
    config.available_disk = st.f_bavail * st.f_frsize

    # Check that there is a reasonable number of samples for joint genotyping
    num_samples = len(samples)
    if config.joint_genotype and not 30 < num_samples < 200:
        job.fileStore.logToMaster('WARNING: GATK recommends batches of '
                                  '30 to 200 samples for joint genotyping. '
                                  'The current cohort has %d samples.' % num_samples)

    shared_files = Job.wrapJobFn(download_shared_files, config).encapsulate()
    job.addChild(shared_files)

    if config.preprocess_only:
        for sample in samples:
            shared_files.addChildJobFn(prepare_bam,
                                       sample.uuid,
                                       sample.url,
                                       shared_files.rv(),
                                       paired_url=sample.paired_url,
                                       rg_line=sample.rg_line)
    else:
        run_pipeline = Job.wrapJobFn(gatk_germline_pipeline,
                                     samples,
                                     shared_files.rv()).encapsulate()
        shared_files.addChild(run_pipeline)

        if config.run_oncotator:
            annotate = Job.wrapJobFn(annotate_vcfs, run_pipeline.rv(), shared_files.rv())
            run_pipeline.addChild(annotate)