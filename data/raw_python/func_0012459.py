def main(args=None):
    """
    Main entry point
    """
    # parse args
    if args is None:
        args = parse_args(sys.argv[1:])

    # set logging level
    if args.verbose > 1:
        set_log_debug()
    elif args.verbose == 1:
        set_log_info()

    outpath = os.path.abspath(os.path.expanduser(args.out_dir))
    cachepath = os.path.abspath(os.path.expanduser(args.cache_dir))
    cache = DiskDataCache(cache_path=cachepath)

    if args.user:
        args.PROJECT = _pypi_get_projects_for_user(args.user)

    if args.query:
        DataQuery(args.project_id, args.PROJECT, cache).run_queries(
            backfill_num_days=args.backfill_days)
    else:
        logger.warning('Query disabled by command-line flag; operating on '
                       'cached data only.')
    if not args.generate:
        logger.warning('Output generation disabled by command-line flag; '
                       'exiting now.')
        raise SystemExit(0)
    for proj in args.PROJECT:
        logger.info('Generating output for: %s', proj)
        stats = ProjectStats(proj, cache)
        outdir = os.path.join(outpath, proj)
        OutputGenerator(proj, stats, outdir).generate()