def omit_deep(omit_props, dct):
    """
    Implementation of omit that recurses. This tests the same keys at every level of dict and in lists
    :param omit_props:
    :param dct:
    :return:
    """

    omit_partial = omit_deep(omit_props)

    if isinstance(dict, dct):
        # Filter out keys and then recurse on each value that wasn't filtered out
        return map_dict(omit_partial, compact_dict(omit(omit_props, dct)))
    if isinstance((list, tuple), dct):
        # run omit_deep on each value
        return map(omit_partial, dct)
    # scalar
    return dct