def map_with_obj(f, dct):
    """
        Implementation of Ramda's mapObjIndexed without the final argument.
        This returns the original key with the mapped value. Use map_key_values to modify the keys too
    :param f: Called with a key and value
    :param dct:
    :return {dict}: Keyed by the original key, valued by the mapped value
    """
    f_dict = {}
    for k, v in dct.items():
        f_dict[k] = f(k, v)
    return f_dict