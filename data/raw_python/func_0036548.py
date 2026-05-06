def list_to_str(list, separator=','):
    """
    >>> list = [0, 0, 7]
    >>> list_to_str(list)
    '0,0,7'
    """
    list = [str(x) for x in list]
    return separator.join(list)