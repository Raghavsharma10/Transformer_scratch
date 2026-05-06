def setup_logging():
    """ Setup logging configuration """

    # Console formatter, mention name
    cfmt = logging.Formatter(('%(name)s - %(levelname)s - %(message)s'))

    # File formatter, mention time
    ffmt = logging.Formatter(('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    # File handler
    fh = logging.handlers.RotatingFileHandler('montblanc.log',
        maxBytes=10*1024*1024, backupCount=10)
    fh.setLevel(logging.INFO)
    fh.setFormatter(ffmt)

    # Create the logger,
    # adding the console and file handler
    mb_logger = logging.getLogger('montblanc')
    mb_logger.handlers = []
    mb_logger.setLevel(logging.DEBUG)
    mb_logger.addHandler(ch)
    mb_logger.addHandler(fh)

    # Set up the concurrent.futures logger
    cf_logger = logging.getLogger('concurrent.futures')
    cf_logger.setLevel(logging.DEBUG)
    cf_logger.addHandler(ch)
    cf_logger.addHandler(fh)

    return mb_logger