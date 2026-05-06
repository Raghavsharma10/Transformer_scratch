def flatten_hierarchical_dict(original_dict, separator='.', max_recursion_depth=None):
    """Flatten a dict.

    Inputs
    ------
    original_dict: dict
        the dictionary to flatten
    separator: string, optional
        the separator item in the keys of the flattened dictionary
    max_recursion_depth: positive integer, optional
        the number of recursions to be done. None is infinte.

    Output
    ------
    the flattened dictionary

    Notes
    -----
    Each element of `original_dict` which is not an instance of `dict` (or of a
    subclass of it) is kept as is. The others are treated as follows. If
    ``original_dict['key_dict']`` is an instance of `dict` (or of a subclass of
    `dict`), a corresponding key of the form
    ``key_dict<separator><key_in_key_dict>`` will be created in
    ``original_dict`` with the value of
    ``original_dict['key_dict']['key_in_key_dict']``.
    If that value is a subclass of `dict` as well, the same procedure is
    repeated until the maximum recursion depth is reached.

    Only string keys are supported.
    """
    if max_recursion_depth is not None and max_recursion_depth <= 0:
        # we reached the maximum recursion depth, refuse to go further
        return original_dict
    if max_recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = max_recursion_depth - 1
    dict1 = {}
    for k in original_dict:
        if not isinstance(original_dict[k], dict):
            dict1[k] = original_dict[k]
        else:
            dict_recursed = flatten_hierarchical_dict(
                original_dict[k], separator, next_recursion_depth)
            dict1.update(
                dict([(k + separator + x, dict_recursed[x]) for x in dict_recursed]))
    return dict1