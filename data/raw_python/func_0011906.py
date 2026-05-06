def isreal(obj):
    """
    Test if the argument is a real number (float or integer).

    :param obj: Object
    :type  obj: any

    :rtype: boolean
    """
    return (
        (obj is not None)
        and (not isinstance(obj, bool))
        and isinstance(obj, (int, float))
    )