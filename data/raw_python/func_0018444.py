def configure_logging(level=logging.DEBUG):
    """Configure the module logging engine."""
    if level == logging.DEBUG:
        # For debugging purposes, log from everyone!
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger