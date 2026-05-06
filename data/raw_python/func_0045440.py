def logSysInfo():
    """Write system info to log file"""
    logger.info('#' * 70)
    logger.info(datetime.today().strftime("%A, %d %B %Y %I:%M%p"))
    logger.info('Running on [{0}] [{1}]'.format(platform.node(),
                                                platform.platform()))
    logger.info('Python [{0}]'.format(sys.version))
    logger.info('#' * 70 + '\n')