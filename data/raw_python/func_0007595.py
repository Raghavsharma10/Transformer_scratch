def set_log_level(debug, verbose):
    """
    Function for setting the logging level.

    :param debug: This boolean field is the logging level.
    :param verbose: This boolean field is the logging level.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        logging.basicConfig(level=logging.INFO)