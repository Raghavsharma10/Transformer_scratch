def enable_log(level=logging.DEBUG):
    """Enable console logging.

    This is a utils method for try run with storops.
    :param level: log level, default to DEBUG
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if not logger.handlers:
        logger.info('enabling logging to console.')
        logger.addHandler(logging.StreamHandler(sys.stdout))