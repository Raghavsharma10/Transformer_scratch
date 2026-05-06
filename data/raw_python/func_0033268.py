def gen_logger(name: str, log_level: int=logging.INFO) -> logging.Logger:
    """Create a logger to be used between processes.

    :returns: Logging instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    shandler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    fmt: str = '\033[1;32m%(levelname)-5s %(module)s:%(funcName)s():'
    fmt += '%(lineno)d %(asctime)s\033[0m| %(message)s'
    shandler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(shandler)
    return logger