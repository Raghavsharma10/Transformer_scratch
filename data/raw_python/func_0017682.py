def download_shared_files(job, input_args):
    """
    Downloads and stores shared inputs files in the FileStore

    input_args: dict        Dictionary of input arguments (from main())
    """
    shared_files = ['unc.bed', 'hg19.transcripts.fa', 'composite_exons.bed', 'normalize.pl', 'rsem_ref.zip',
                    'ebwt.zip', 'chromosomes.zip']
    shared_ids = {}
    for f in shared_files:
        shared_ids[f] = job.addChildJobFn(download_from_url, input_args[f]).rv()
    if input_args['config'] or input_args['config_fastq']:
        job.addFollowOnJobFn(parse_config_file, shared_ids, input_args)
    else:
        sample_path = input_args['input']
        uuid = os.path.splitext(os.path.basename(sample_path))[0]
        sample = (uuid, sample_path)
        job.addFollowOnJobFn(download_sample, shared_ids, input_args, sample)