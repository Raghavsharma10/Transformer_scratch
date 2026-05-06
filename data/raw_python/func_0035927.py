def set_level(logger, level):
    '''
    Temporarily change log level of logger.

    Parameters
    ----------
    logger : str or ~logging.Logger
        Logger name or logger whose log level to change.
    level : int
        Log level to set.

    Examples
    --------
    >>> with set_level('sqlalchemy.engine', logging.INFO):
    ...     pass  # sqlalchemy log level is set to INFO in this block
    '''
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    original = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original)