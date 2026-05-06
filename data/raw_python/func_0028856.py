def map_prop_value_as_index(prp, lst):
    """
        Returns the given prop of each item in the list
    :param prp:
    :param lst:
    :return:
    """
    return from_pairs(map(lambda item: (prop(prp, item), item), lst))