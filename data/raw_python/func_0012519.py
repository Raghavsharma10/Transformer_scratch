def remove_all_handlers(logger=None):
    """
    Safely remove all handlers from the logger

    :param logger: logging name or object to modify, defaults to root logger
    """
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    remove_file_handlers(logger)
    logger.handlers = []