def from_pairs_to_array_values(pairs):
    """
        Like from pairs but combines duplicate key values into arrays
    :param pairs:
    :return:
    """
    result = {}
    for pair in pairs:
        result[pair[0]] = concat(prop_or([], pair[0], result), [pair[1]])
    return result