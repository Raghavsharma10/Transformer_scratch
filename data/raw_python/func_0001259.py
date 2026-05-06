def config_logging(args):
    ''' Override root logger's level '''
    if args.quiet:
        logging.getLogger().setLevel(logging.CRITICAL)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)