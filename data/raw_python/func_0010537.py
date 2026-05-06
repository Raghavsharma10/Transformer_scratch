def logging_file_install(path):
    """
    Install logger that will write to file. If this function has already installed a handler, replace it.
    :param path: path to the log file, Use None for default file location.
    """
    if path is None:
        path = configuration_get_default_folder() / LOGGING_DEFAULTNAME

    if not path.parent.exists():
        log.error('File logger installation FAILED!')
        log.error('The directory of the log file does not exist.')
        return

    formatter = logging.Formatter(LOGGING_FORMAT)
    logger = logging.getLogger()

    logger.removeHandler(LOGGING_HANDLERS['file'])

    logFileHandler = logging.handlers.RotatingFileHandler(filename=str(path),
                                                          mode='a',
                                                          maxBytes=LOGGING_MAXBYTES,
                                                          backupCount=LOGGING_BACKUPCOUNT)
    logFileHandler.setLevel(logging.DEBUG)
    logFileHandler.setFormatter(formatter)

    LOGGING_HANDLERS['file'] = logFileHandler

    logger.addHandler(logFileHandler)