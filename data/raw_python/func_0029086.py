def getLogger(name):
    """This is used by gcdt plugins to get a logger with the right level."""
    logger = logging.getLogger(name)
    # note: the level might be adjusted via '-v' option
    logger.setLevel(logging_config['loggers']['gcdt']['level'])
    return logger