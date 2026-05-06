def main():
    """
    Transfer gTEX data from dbGaP (NCBI) to S3
    """
    # Define Parser object and add to toil
    parser = build_parser()
    Job.Runner.addToilOptions(parser)
    args = parser.parse_args()
    # Store inputs from argparse
    inputs = {'sra': args.sra,
              'dbgap_key': args.dbgap_key,
              'ssec': args.ssec,
              's3_dir': args.s3_dir,
              'single_end': args.single_end,
              'sudo': args.sudo}
    # Sanity checks
    if args.ssec:
        assert os.path.isfile(args.ssec)
    if args.sra:
        assert os.path.isfile(args.sra)
    if args.dbgap_key:
        assert os.path.isfile(args.dbgap_key)
    # Start Pipeline
    Job.Runner.startToil(Job.wrapJobFn(start_batch, inputs), args)