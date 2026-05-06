def _create_logger(name: str, level: int) -> Generator[logging.Logger, None, None]:
    """Create a context-based logger.
    Parameters
    ----------
    name: str
        Name of logger to use when logging.
    level: int
        Logging level, one of logging's levels (e.g. INFO, ERROR, etc.).

    Returns
    -------
    logging.Logger
        Named logger that may be used for logging.
    """
    # Get logger
    logger = logging.getLogger(name)

    # Set logger level
    old_level = logger.level
    logger.setLevel(level)

    # Setup handler and add to logger
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s [%(name)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    yield logger

    # Reset logger level
    logger.setLevel(old_level)

    # Remove handler from logger
    logger.removeHandler(handler)
    handler.close()