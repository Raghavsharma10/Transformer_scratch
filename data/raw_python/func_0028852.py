def map_keys_with_obj(f, dct):
    """
        Calls f with each key and value of dct, possibly returning a modified key. Values are unchanged
    :param f: Called with each key and value and returns the same key or a modified key
    :param dct:
    :return: A dct with keys possibly modifed but values unchanged
    """
    f_dict = {}
    for k, v in dct.items():
        f_dict[f(k, v)] = v
    return f_dict