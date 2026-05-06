def remove_stream_handlers(logger=None):
    """
    Remove only stream handlers from the specified logger

    :param logger: logging name or object to modify, defaults to root logger
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    new_handlers = []
    for handler in logger.handlers:
        # FileHandler is a subclass of StreamHandler so
        # 'if not a StreamHandler' does not work
        if (isinstance(handler, logging.FileHandler) or
            isinstance(handler, logging.NullHandler) or
            (isinstance(handler, logging.Handler) and not
                isinstance(handler, logging.StreamHandler))):
            new_handlers.append(handler)
    logger.handlers = new_handlers