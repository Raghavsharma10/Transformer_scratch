def variant_to_canonical_string(obj):
    """
    Return a list containing the canonical string for the given object.

    The ``obj`` can be a list or a set of descriptor strings, or a Unicode string.
    
    If ``obj`` is a Unicode string, it will be split using spaces as delimiters.

    :param variant obj: the object to be parsed
    :rtype: str 
    :raise TypeError: if the ``obj`` has a type not listed above
    """
    acc = [DG_ALL_DESCRIPTORS.canonical_value(p) for p in variant_to_list(obj)]
    acc = sorted([a for a in acc if a is not None])
    return u" ".join(acc)