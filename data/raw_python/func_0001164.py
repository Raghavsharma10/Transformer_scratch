def entry_point():
    """Entry-point from setuptools."""
    signal.signal(signal.SIGINT, lambda *_: getattr(os, '_exit')(0))  # Properly handle Control+C
    config = get_arguments()
    setup_logging(config['verbose'])
    try:
        main(config)
    except HandledError:
        if config['raise']:
            raise
        logging.critical('Failure.')
        sys.exit(0 if config['ignore_errors'] else 1)