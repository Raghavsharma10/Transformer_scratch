def add_stream_handler(logger=None, stream=sys.stderr, level=logging.INFO,
                       log_format=log_formats.easy_read):
    """
    Addes a newly created stream handler to the specified logger

    :param logger: logging name or object to modify, defaults to root logger
    :param stream: which stream to use, defaults to sys.stderr
    :param level: logging level to set handler at
    :param log_format: formatter to use
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    logger.addHandler(get_stream_handler(stream, level, log_format))