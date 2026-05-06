def get_stream_handler(stream=sys.stderr, level=logging.INFO,
                       log_format=log_formats.easy_read):
    """
    Returns a set up stream handler to add to a logger.

    :param stream: which stream to use, defaults to sys.stderr
    :param level: logging level to set handler at
    :param log_format: formatter to use
    :return: stream handler
    """
    sh = logging.StreamHandler(stream)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(log_format))
    return sh