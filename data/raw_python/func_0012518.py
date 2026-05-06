def remove_file_handlers(logger=None):
    """
    Remove only file handlers from the specified logger. Will go through
    and close each handler for safety.

    :param logger: logging name or object to modify, defaults to root logger
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    new_handlers = []
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        else:
            new_handlers.append(handler)
    logger.handlers = new_handlers