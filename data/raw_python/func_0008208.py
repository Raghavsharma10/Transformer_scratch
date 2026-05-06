def get_logger_by_name(name=None, rand_name=False, charset=Charset.HEX):
    """
    Get a logger by name.

    :param name: None / str, logger name.
    :param rand_name: if True, ``name`` will be ignored, a random name will be used.
    """
    if rand_name:
        name = rand_str(charset)
    logger = logging.getLogger(name)
    return logger