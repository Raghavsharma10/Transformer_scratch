def change_logger_levels(logger=None, level=logging.DEBUG):
    """
    Go through the logger and handlers and update their levels to the
    one specified.

    :param logger: logging name or object to modify, defaults to root logger
    :param level: logging level to set at (10=Debug, 20=Info, 30=Warn, 40=Error)
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.level = level