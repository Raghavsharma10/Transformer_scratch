def initialize_logging(args):
    """Configure the root logger with some sensible defaults."""
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        "%(levelname)s %(asctime)s %(name)s:%(lineno)04d - %(message)s")
    log_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(getattr(logging, args.loglevel))