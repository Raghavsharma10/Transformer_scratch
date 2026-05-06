def diff_sizes(a, b, progressbar=None):
    """Return list of tuples where sizes differ.

    Tuple structure:
    (identifier, size in a, size in b)

    Assumes list of identifiers in a and b are identical.

    :param a: first :class:`dtoolcore.DataSet`
    :param b: second :class:`dtoolcore.DataSet`
    :returns: list of tuples for all items with different sizes
    """
    difference = []

    for i in a.identifiers:
        a_size = a.item_properties(i)["size_in_bytes"]
        b_size = b.item_properties(i)["size_in_bytes"]
        if a_size != b_size:
            difference.append((i, a_size, b_size))
        if progressbar:
            progressbar.update(1)

    return difference