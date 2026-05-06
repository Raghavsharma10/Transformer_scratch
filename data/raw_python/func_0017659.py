def static_dag(job, uuid, rg_line, inputs):
    """
    Prefer this here as it allows us to pull the job functions from other jobs
    without rewrapping the job functions back together.

    bwa_inputs: Input arguments to be passed to BWA.
    adam_inputs: Input arguments to be passed to ADAM.
    gatk_preprocess_inputs: Input arguments to be passed to GATK preprocessing.
    gatk_adam_call_inputs: Input arguments to be passed to GATK haplotype caller for the result of ADAM preprocessing.
    gatk_gatk_call_inputs: Input arguments to be passed to GATK haplotype caller for the result of GATK preprocessing.
    """

    # get work directory
    work_dir = job.fileStore.getLocalTempDir()

    inputs.cpu_count = cpu_count()
    inputs.maxCores = sys.maxint
    args = {'uuid': uuid,
            's3_bucket': inputs.s3_bucket,
            'sequence_dir': inputs.sequence_dir,
            'dir_suffix': inputs.dir_suffix}

    # get head BWA alignment job function and encapsulate it
    inputs.rg_line = rg_line
    inputs.output_dir = 's3://{s3_bucket}/alignment{dir_suffix}'.format(**args)
    bwa = job.wrapJobFn(download_reference_files,
                        inputs,
                        [[uuid,
                         ['s3://{s3_bucket}/{sequence_dir}/{uuid}_1.fastq.gz'.format(**args),
                          's3://{s3_bucket}/{sequence_dir}/{uuid}_2.fastq.gz'.format(**args)]]]).encapsulate()

    # get head ADAM preprocessing job function and encapsulate it
    adam_preprocess = job.wrapJobFn(static_adam_preprocessing_dag,
                                    inputs,
                                    's3://{s3_bucket}/alignment{dir_suffix}/{uuid}.bam'.format(**args),
                                    's3://{s3_bucket}/analysis{dir_suffix}/{uuid}'.format(**args),
                                    suffix='.adam').encapsulate()

    # Configure options for Toil Germline pipeline. This function call only runs the preprocessing steps.
    gatk_preprocessing_inputs = copy.deepcopy(inputs)
    gatk_preprocessing_inputs.suffix = '.gatk'
    gatk_preprocessing_inputs.preprocess = True
    gatk_preprocessing_inputs.preprocess_only = True
    gatk_preprocessing_inputs.output_dir = 's3://{s3_bucket}/analysis{dir_suffix}'.format(**args)

    # get head GATK preprocessing job function and encapsulate it
    gatk_preprocess = job.wrapJobFn(run_gatk_germline_pipeline,
                                    GermlineSample(uuid,
                                                   's3://{s3_bucket}/alignment{dir_suffix}/{uuid}.bam'.format(**args),
                                                   None,    # Does not require second URL or RG_Line
                                                   None),
                                    gatk_preprocessing_inputs).encapsulate()

    # Configure options for Toil Germline pipeline for preprocessed ADAM BAM file.
    adam_call_inputs = inputs
    adam_call_inputs.suffix = '.adam'
    adam_call_inputs.sorted = True
    adam_call_inputs.preprocess = False
    adam_call_inputs.run_vqsr = False
    adam_call_inputs.joint_genotype = False
    adam_call_inputs.output_dir = 's3://{s3_bucket}/analysis{dir_suffix}'.format(**args)

    # get head GATK haplotype caller job function for the result of ADAM preprocessing and encapsulate it
    gatk_adam_call = job.wrapJobFn(run_gatk_germline_pipeline,
                                   GermlineSample(uuid,
                                                  's3://{s3_bucket}/analysis{dir_suffix}/{uuid}/{uuid}.adam.bam'.format(**args),
                                                  None,
                                                  None),
                                   adam_call_inputs).encapsulate()

    # Configure options for Toil Germline pipeline for preprocessed GATK BAM file.
    gatk_call_inputs = copy.deepcopy(inputs)
    gatk_call_inputs.sorted = True
    gatk_call_inputs.preprocess = False
    gatk_call_inputs.run_vqsr = False
    gatk_call_inputs.joint_genotype = False
    gatk_call_inputs.output_dir = 's3://{s3_bucket}/analysis{dir_suffix}'.format(**args)

    # get head GATK haplotype caller job function for the result of GATK preprocessing and encapsulate it
    gatk_gatk_call = job.wrapJobFn(run_gatk_germline_pipeline,
                                   GermlineSample(uuid,
                                                  'S3://{s3_bucket}/analysis{dir_suffix}/{uuid}/{uuid}.gatk.bam'.format(**args),
                                                  None, None),
                                   gatk_call_inputs).encapsulate()

    # wire up dag
    if not inputs.skip_alignment:
        job.addChild(bwa)

    if (inputs.pipeline_to_run == "adam" or
        inputs.pipeline_to_run == "both"):

        if inputs.skip_preprocessing:

            job.addChild(gatk_adam_call)
        else:
            if inputs.skip_alignment:
                job.addChild(adam_preprocess)
            else:
                bwa.addChild(adam_preprocess)

            adam_preprocess.addChild(gatk_adam_call)

    if (inputs.pipeline_to_run == "gatk" or
        inputs.pipeline_to_run == "both"):

        if inputs.skip_preprocessing:

            job.addChild(gatk_gatk_call)
        else:
            if inputs.skip_alignment:
                job.addChild(gatk_preprocess)
            else:
                bwa.addChild(gatk_preprocess)

            gatk_preprocess.addChild(gatk_gatk_call)