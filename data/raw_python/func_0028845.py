def map_with_obj_deep(f, dct):
    """
    Implementation of map that recurses. This tests the same keys at every level of dict and in lists
    :param f: 2-ary function expecting a key and value and returns a modified value
    :param dct: Dict for deep processing
    :return: Modified dct with matching props mapped
    """
    return _map_deep(lambda k, v: [k, f(k, v)], dct)