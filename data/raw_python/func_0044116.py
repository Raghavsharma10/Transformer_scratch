def setup_logging(logging_config, debug=False):
    """Setup logging config."""

    if logging_config is not None:
        logging.config.fileConfig(logging_config)

    else:
        logging.basicConfig(level=debug and logging.DEBUG or logging.ERROR)