def is_dict_equal(d1, d2, keys=None, ignore_none_values=True):
    """
    Compares two dictionaries to see if they are equal
    :param d1: the first dictionary
    :param d2: the second dictionary
    :param keys: the keys to limit the comparison to (optional)
    :param ignore_none_values: whether to ignore none values
    :return: true if the dictionaries are equal, else false
    """
    if keys or ignore_none_values:
        d1 = {k: v for k, v in d1.items() if (keys is None or k in keys) and (v is not None or not ignore_none_values)}
        d2 = {k: v for k, v in d2.items() if (keys is None or k in keys) and (v is not None or not ignore_none_values)}

    return d1 == d2