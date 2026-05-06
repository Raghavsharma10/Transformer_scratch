def parse_input_samples(job, inputs):
    """
    Parses config file to pull sample information.
    Stores samples as tuples of (uuid, URL)

    :param JobFunctionWrappingJob job: passed by Toil automatically
    :param Namespace inputs: Stores input arguments (see main)
    """
    job.fileStore.logToMaster('Parsing input samples and batching jobs')
    samples = []
    if inputs.config:
        with open(inputs.config, 'r') as f:
            for line in f.readlines():
                if not line.isspace():
                    sample = line.strip().split(',')
                    assert len(sample) == 2, 'Error: Config file is inappropriately formatted.'
                    samples.append(sample)
    job.addChildJobFn(map_job, download_sample, samples, inputs)