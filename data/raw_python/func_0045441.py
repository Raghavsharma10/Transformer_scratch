def logEndTime():
    """Write end info to log"""
    logger.info('\n' + '#' * 70)
    logger.info('Complete')
    logger.info(datetime.today().strftime("%A, %d %B %Y %I:%M%p"))
    logger.info('#' * 70 + '\n')