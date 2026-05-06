def get_handler_fp(logger):
    """
    Get handler_fp.
    This method is integrated to LoggerFactory Object in the future.
    :param logging.Logger logger: Python logging.Logger. logger instance.
    :rtype: logging.Logger.handlers.BaseRotatingHandler
    :return: Handler or Handler's stream. We call it `handler_fp`.
    """
    if not hasattr(logger, 'handlers'):
        raise blackbird.utils.error.BlackbirdError(
            'Given logger is not logging.Logger instance!'
        )

    if len(logger.handlers) != 1:
        raise blackbird.utils.error.BlackbirdError(
            'Given logger has invalid handlers.'
        )

    if hasattr(logger.handlers[0], 'stream'):
        return logger.handlers[0].stream

    # case of setting SysLogHandler to logger.handlers[0]
    return logger.handlers[0]