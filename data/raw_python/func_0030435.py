def get_logger(name, file_name=None, stream=None, template=None, propagate=False, level=None):
    """Get a logger by name.

    """

    logger = logging.getLogger(name)
    running_tests = (
        'test' in sys.argv  # running with setup.py
        or sys.argv[0].endswith('py.test'))  # running with py.test
    if running_tests and not level:
        # testing without level, this means tester does not want to see any log messages.
        level = logging.CRITICAL

    if not level:
        level = logging.INFO
    logger.setLevel(level)
    logger.propagate = propagate

    formatter = logging.Formatter(template)

    if not stream:
        stream = sys.stdout

    logger.handlers = []
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if file_name:
        handler = logging.FileHandler(file_name)
        handler.setFormatter(logging.Formatter('%(asctime)s '+template))
        logger.addHandler(handler)

    return logger