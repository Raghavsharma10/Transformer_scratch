def variant_to_list(obj):
    """
    Return a list containing the descriptors in the given object.

    The ``obj`` can be a list or a set of descriptor strings, or a Unicode string.
    
    If ``obj`` is a Unicode string, it will be split using spaces as delimiters.

    :param variant obj: the object to be parsed
    :rtype: list 
    :raise TypeError: if the ``obj`` has a type not listed above
    """
    if isinstance(obj, list):
        return obj
    elif is_unicode_string(obj):
        return [s for s in obj.split() if len(s) > 0]
    elif isinstance(obj, set) or isinstance(obj, frozenset):
        return list(obj)
    raise TypeError("The given value must be a list or a set of descriptor strings, or a Unicode string.")