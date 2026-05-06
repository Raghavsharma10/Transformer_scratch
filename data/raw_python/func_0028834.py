def prop(key, dct_or_obj):
    """
        Implementation of prop (get_item) that also supports object attributes
    :param key:
    :param dct_or_obj:
    :return:
    """
    # Note that hasattr is a builtin and getattr is a ramda function, hence the different arg position
    if isinstance(dict, dct_or_obj):
        if has(key, dct_or_obj):
            return dct_or_obj[key]
        else:
            raise Exception("No key %s found for dict %s" % (key, dct_or_obj))
    elif isinstance(list, dct_or_obj):
        if isint(key):
            return dct_or_obj[key]
        else:
            raise Exception("Key %s not expected for list type: %s" % (key, dct_or_obj))
    elif isinstance(object, dct_or_obj):
        if hasattr(dct_or_obj, key):
            return getattr(key, dct_or_obj)
        else:
            raise Exception("No key %s found for objects %s" % (key, dct_or_obj))
    else:
        raise Exception("%s is neither a dict nor objects" % dct_or_obj)