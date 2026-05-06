def parse_config_file(job, ids, input_args):
    """
    Launches pipeline for each sample.

    shared_ids: dict        Dictionary of fileStore IDs
    input_args: dict        Dictionary of input arguments
    """
    samples = []
    config = input_args['config']
    with open(config, 'r') as f:
        for line in f.readlines():
            if not line.isspace():
                sample = line.strip().split(',')
                samples.append(sample)
    for sample in samples:
        job.addChildJobFn(download_sample, ids, input_args, sample)