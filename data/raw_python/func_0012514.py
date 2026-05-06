def add_file_handler(logger=None, file_path="out.log", level=logging.INFO,
                     log_format=log_formats.easy_read):
    """
    Addes a newly created file handler to the specified logger

    :param logger: logging name or object to modify, defaults to root logger
    :param file_path: path to file to log to
    :param level: logging level to set handler at
    :param log_format: formatter to use
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    logger.addHandler(get_file_handler(file_path, level, log_format))