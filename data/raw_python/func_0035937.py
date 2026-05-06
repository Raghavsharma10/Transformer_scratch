def diff_identifiers(a, b):
    """Return list of tuples where identifiers in datasets differ.

    Tuple structure:
    (identifier, present in a, present in b)

    :param a: first :class:`dtoolcore.DataSet`
    :param b: second :class:`dtoolcore.DataSet`
    :returns: list of tuples where identifiers in datasets differ
    """

    a_ids = set(a.identifiers)
    b_ids = set(b.identifiers)

    difference = []

    for i in a_ids.difference(b_ids):
        difference.append((i, True, False))
    for i in b_ids.difference(a_ids):
        difference.append((i, False, True))

    return difference