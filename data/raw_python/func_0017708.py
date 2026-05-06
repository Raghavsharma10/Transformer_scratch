def main():
    """
    This is a Toil pipeline to transfer TCGA data into an S3 Bucket

    Data is pulled down with Genetorrent and transferred to S3 via S3AM.
    """
    # Define Parser object and add to toil
    parser = build_parser()
    Job.Runner.addToilOptions(parser)
    args = parser.parse_args()
    # Store inputs from argparse
    inputs = {'genetorrent': args.genetorrent,
              'genetorrent_key': args.genetorrent_key,
              'ssec': args.ssec,
              's3_dir': args.s3_dir}
    # Sanity checks
    if args.ssec:
        assert os.path.isfile(args.ssec)
    if args.genetorrent:
        assert os.path.isfile(args.genetorrent)
    if args.genetorrent_key:
        assert os.path.isfile(args.genetorrent_key)
    samples = parse_genetorrent(args.genetorrent)
    # Start pipeline
    # map_job accepts a function, an iterable, and *args. The function is launched as a child
    # process with one element from the iterable and *args, which in turn spawns a tree of child jobs.
    Job.Runner.startToil(Job.wrapJobFn(map_job, download_and_transfer_sample, samples, inputs), args)