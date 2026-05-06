def pick_deep(pick_dct, dct):
    """
    Implementation of pick that recurses. This tests the same keys at every level of dict and in lists
    :param pick_dct: Deep dict matching some portion of dct.
    :param dct: Dct to filter. Any key matching pick_dct pass through. It doesn't matter what the pick_dct value
    is as long as the key exists. Arrays also pass through if the have matching values in pick_dct
    :return:
    """

    if isinstance(dict, dct):
        # Filter out keys and then recurse on each value that wasn't filtered out
        return map_with_obj(
            lambda k, v: pick_deep(prop(k, pick_dct), v),
            pick(keys(pick_dct), dct)
        )
    if isinstance((list, tuple), dct):
        # run pick_deep on each value
        return map(
            lambda tup: pick_deep(*tup),
            list(zip(pick_dct or [], dct))
        )
    # scalar
    return dct