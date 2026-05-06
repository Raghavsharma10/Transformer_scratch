def spawn_batch_jobs(job, shared_ids, input_args):
    """
    Spawns an alignment job for every sample in the input configuration file
    """
    samples = []
    config = input_args['config']
    with open(config, 'r') as f_in:
        for line in f_in:
            line = line.strip().split(',')
            uuid = line[0]
            urls = line[1:]
            samples.append((uuid, urls))
    for sample in samples:
        job.addChildJobFn(alignment, shared_ids, input_args, sample, cores=32, memory='20 G', disk='100 G')