def add_log_level(value, name):
    """
    Add a new log level to the :mod:`logging` module.

    :param value: The log level's number (an integer).
    :param name: The name for the log level (a string).
    """
    logging.addLevelName(value, name)
    setattr(logging, name, value)