def _map_deep(f, dct):
    """
    Used by map_deep and map_keys_deep
    :param map_props:
    :param f: Expects a key and value and returns a pair
    :param dct:
    :return:
    """

    if isinstance(dict, dct):
        return map_key_values(lambda k, v: f(k, _map_deep(f, v)), dct)
    elif isinstance((list, tuple), dct):
        # Call each value with the index as the key. Since f returns a key value discard the key that it returns
        # Even if this is called with map_keys_deep we can't manipulate index values here
        return map(lambda iv: f(iv[0], _map_deep(f, iv[1]))[1], enumerate(dct))
    # scalar
    return dct