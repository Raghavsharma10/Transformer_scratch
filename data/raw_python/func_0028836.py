def prop_or(default, key, dct_or_obj):
    """
        Ramda propOr implementation. This also resolves object attributes, so key
        can be a dict prop or an attribute of dct_or_obj
    :param default: Value if dct_or_obj doesn't have key_or_prop or the resolved value is null
    :param key:
    :param dct_or_obj:
    :return:
    """
    # Note that hasattr is a builtin and getattr is a ramda function, hence the different arg position
    if isinstance(dict, dct_or_obj):
        value = dct_or_obj[key] if has(key, dct_or_obj) else default
    elif isinstance(object, dct_or_obj):
        value = getattr(key, dct_or_obj) if hasattr(dct_or_obj, key) else default
    else:
        value = default
    # 0 and False are ok, None defaults
    if value == None:
        return default
    return value