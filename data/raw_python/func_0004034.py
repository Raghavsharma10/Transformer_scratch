def is_list_of_ipachars(obj):
    """
    Return ``True`` if the given object is a list of IPAChar objects.

    :param object obj: the object to test
    :rtype: bool
    """
    if isinstance(obj, list):
        for e in obj:
            if not isinstance(e, IPAChar):
                return False
        return True
    return False