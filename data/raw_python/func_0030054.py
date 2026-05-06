def set_log_level(level):
    """Sets the desired log level."""
    lLevel = level.lower()
    unrecognized = False
    if (lLevel == 'debug-all'):
        loglevel = logging.DEBUG
    elif (lLevel == 'debug'):
        loglevel = logging.DEBUG
    elif (lLevel == 'info'):
        loglevel = logging.INFO
    elif (lLevel == 'warning'):
        loglevel = logging.WARNING
    elif (lLevel == 'error'):
        loglevel = logging.ERROR
    elif (lLevel == 'critical'):
        loglevel = logging.CRITICAL
    else:
        loglevel = logging.DEBUG
        unrecognized = True
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d/%(funcName)s: %(message)s')
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').setLevel(loglevel)
    #logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d/%(funcName)s: %(message)s', level=loglevel)
    if lLevel != 'debug-all':
        # lower the loglevel for enumerated packages to avoid unwanted messages
        packagesWarning = ["requests.packages.urllib3", "urllib3", "requests_kerberos", "jenkinsapi"]
        for package in packagesWarning:
            logging.debug("Setting loglevel for %s to WARNING.", package)
            logger = logging.getLogger(package)
            logger.setLevel(logging.WARNING)

    if unrecognized:
        logging.warning('Unrecognized log level: %s  Log level set to debug', level)

    #TODO ref: use external log config
    fh = logging.FileHandler('builder.log')
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)
    logging.getLogger('').addHandler(fh)