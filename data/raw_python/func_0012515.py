def add_rotating_file_handler(logger=None, file_path="out.log",
                              level=logging.INFO,
                              log_format=log_formats.easy_read,
                              max_bytes=10*sizes.mb, backup_count=5,
                              **handler_kwargs):
    """ Adds a rotating file handler to the specified logger.

    :param logger: logging name or object to modify, defaults to root logger
    :param file_path: path to file to log to
    :param level: logging level to set handler at
    :param log_format: log formatter
    :param max_bytes: Max file size in bytes before rotating
    :param backup_count: Number of backup files
    :param handler_kwargs: options to pass to the handler
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    logger.addHandler(get_file_handler(file_path, level, log_format,
                                       handler=RotatingFileHandler,
                                       maxBytes=max_bytes,
                                       backupCount=backup_count,
                                       **handler_kwargs))