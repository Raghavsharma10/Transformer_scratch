def debug_setup(logger=None, level=None, log2file=None,
                log_file=None, log_format=None, log_dir=None,
                log2stdout=None, truncate=False):
    '''
    Local object instance logger setup.

    Verbosity levels are determined as such::

        if level in [-1, False]:
            logger.setLevel(logging.WARN)
        elif level in [0, None]:
            logger.setLevel(logging.INFO)
        elif level in [True, 1, 2]:
            logger.setLevel(logging.DEBUG)

    If (level == 2) `logging.DEBUG` will be set even for
    the "root logger".

    Configuration options available for customized logger behaivor:
        * debug (bool)
        * log2stdout (bool)
        * log2file (bool)
        * log_file (path)
    '''
    log2stdout = False if log2stdout is None else log2stdout
    _log_format = "%(levelname)s.%(name)s.%(process)s:%(asctime)s:%(message)s"
    log_format = log_format or _log_format
    if isinstance(log_format, basestring):
        log_format = logging.Formatter(log_format, "%Y%m%dT%H%M%S")

    log2file = True if log2file is None else log2file
    logger = logger or 'metrique'
    if isinstance(logger, basestring):
        logger = logging.getLogger(logger)
    else:
        logger = logger or logging.getLogger(logger)
    logger.propagate = 0
    logger.handlers = []
    if log2file:
        log_dir = log_dir or LOGS_DIR
        log_file = log_file or 'metrique'
        log_file = os.path.join(log_dir, log_file)
        if truncate:
            # clear the existing data before writing (truncate)
            open(log_file, 'w+').close()
        hdlr = logging.FileHandler(log_file)
        hdlr.setFormatter(log_format)
        logger.addHandler(hdlr)
    else:
        log2stdout = True
    if log2stdout:
        hdlr = logging.StreamHandler()
        hdlr.setFormatter(log_format)
        logger.addHandler(hdlr)
    logger = _debug_set_level(logger, level)
    return logger