def get_file_handler(file_path="out.log", level=logging.INFO,
                     log_format=log_formats.easy_read,
                     handler=logging.FileHandler,
                     **handler_kwargs):
    """
    Set up a file handler to add to a logger.

    :param file_path: file to write the log to, defaults to out.log
    :param level: logging level to set handler at
    :param log_format: formatter to use
    :param handler: logging handler to use, defaults to FileHandler
    :param handler_kwargs: options to pass to the handler
    :return: handler
    """
    fh = handler(file_path, **handler_kwargs)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(log_format))
    return fh