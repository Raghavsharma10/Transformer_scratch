def run(configuration: str, level: str, target: str, short_format: bool):
    """Run the daemon and all its services"""
    initialise_logging(level=level, target=target, short_format=short_format)
    logger = logging.getLogger(__package__)
    logger.info('COBalD %s', cobald.__about__.__version__)
    logger.info(cobald.__about__.__url__)
    logger.info('%s %s (%s)', platform.python_implementation(), platform.python_version(), sys.executable)
    logger.debug(cobald.__file__)
    logger.info('Using configuration %s', configuration)
    with load(configuration):
        logger.info('Starting daemon services...')
        runtime.accept()