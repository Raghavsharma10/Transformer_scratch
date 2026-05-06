def map_keys(f, dct):
    """
        Calls f with each key of dct, possibly returning a modified key. Values are unchanged
    :param f: Called with each key and returns the same key or a modified key
    :param dct:
    :return: A dct with keys possibly modifed but values unchanged
    """
    f_dict = {}
    for k, v in dct.items():
        f_dict[f(k)] = v
    return f_dict