def ishex(obj):
    """
    Test if the argument is a string representing a valid hexadecimal digit.

    :param obj: Object
    :type  obj: any

    :rtype: boolean
    """
    return isinstance(obj, str) and (len(obj) == 1) and (obj in string.hexdigits)