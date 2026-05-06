def setup_logger(module_name=None, level=logging.INFO, stream=sys.stderr,
                 file_path=None, log_format=log_formats.easy_read,
                 suppress_warning=True):
    """
    Grabs the specified logger and adds wanted handlers to it. Will
    default to adding a stream handler.

    :param module_name: logger name to use
    :param level: logging level to set logger at
    :param stream: stream to log to, or None
    :param file_path: file path to log to, or None
    :param log_format: format to set the handlers to use
    :param suppress_warning: add a NullHandler if no other handler is specified
    :return: configured logger
    """
    new_logger = logging.getLogger(module_name)

    if stream:
        new_logger.addHandler(get_stream_handler(stream, level, log_format))
    elif not file_path and suppress_warning and not new_logger.handlers:
            new_logger.addHandler(logging.NullHandler())

    if file_path:
        new_logger.addHandler(get_file_handler(file_path, level, log_format))
    if level > 0:
        new_logger.setLevel(level)
    return new_logger