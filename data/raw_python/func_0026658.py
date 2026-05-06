def load_log(args):
    """Load a `logging.Logger` object.

    Arguments
    ---------
    args : `argparse.Namespace` object
        Namespace containing required settings:
        {`args.debug`, `args.verbose`, and `args.log_filename`}.

    Returns
    -------
    log : `logging.Logger` object

    """
    from astrocats.catalog.utils import logger

    # Determine verbosity ('None' means use default)
    log_stream_level = None
    if args.debug:
        log_stream_level = logger.DEBUG
    elif args.verbose:
        log_stream_level = logger.INFO

    # Create log
    log = logger.get_logger(
        stream_level=log_stream_level, tofile=args.log_filename)
    log._verbose = args.verbose
    log._debug = args.debug
    return log