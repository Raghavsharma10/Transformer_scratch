def get_level(level_string):
    """
    Returns an appropriate logging level integer from a string name
    """
    levels = {'debug': logging.DEBUG, 'info': logging.INFO,
              'warning': logging.WARNING, 'error': logging.ERROR,
              'critical': logging.CRITICAL}
    try:
        level = levels[level_string.lower()]
    except KeyError:
        sys.exit('{0} is not a recognized logging level'.format(level_string))
    else:
        return level