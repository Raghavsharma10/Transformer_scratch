def log(message: str, *args: str, category: str='info', logger_name: str='pgevents'):
    """Log a message to the given logger.

    If debug has not been enabled, this method will not log a message.

    Parameters
    ----------
    message: str
        Message, with or without formatters, to print.
    args: Any
        Arguments to use with the message. args must either be a series of
        arguments that match up with anonymous formatters
        (i.e. "%<FORMAT-CHARACTER>") in the format string, or a dictionary
        with key-value pairs that match up with named formatters
        (i.e. "%(key)s") in the format string.
    logger_name: str
        Name of logger to which the message should be logged.

    """
    global _DEBUG_ENABLED

    if _DEBUG_ENABLED:
        level = logging.INFO
    else:
        level = logging.CRITICAL + 1

    with _create_logger(logger_name, level) as logger:
        log_fn = getattr(logger, category, None)
        if log_fn is None:
            raise ValueError('Invalid log category "{}"'.format(category))

        log_fn(message, *args)