def toFloat(value):
    """ Converts string or signed list to float. """
    if isinstance(value, str):
        return strFloat(value)
    elif isinstance(value, list):
        return slistFloat(value)
    else:
        return value