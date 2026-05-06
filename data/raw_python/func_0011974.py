def _isreal(obj):
    """
    Determine if an object is a real number.

    Both Python standard data types and Numpy data types are supported.

    :param obj: Object
    :type  obj: any

    :rtype: boolean
    """
    # pylint: disable=W0702
    if (obj is None) or isinstance(obj, bool):
        return False
    try:
        cond = (int(obj) == obj) or (float(obj) == obj)
    except:
        return False
    return cond