def convert(to, who, default=u''):
    """
    Converts data to a specific datatype.
    In case of error, will return a default value.

    :param to:      datatype to be casted to.
    :param who:     value to cast.
    :param default: value to return in case of error.
    :return: a str value.
    """
    if who is None:
        return default
    try:
        return to(who)
    except:  # noqa
        return default