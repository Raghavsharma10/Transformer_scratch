def diff_content(a, reference, progressbar=None):
    """Return list of tuples where content differ.

    Tuple structure:
    (identifier, hash in a, hash in reference)

    Assumes list of identifiers in a and b are identical.

    Storage broker of reference used to generate hash for files in a.

    :param a: first :class:`dtoolcore.DataSet`
    :param b: second :class:`dtoolcore.DataSet`
    :returns: list of tuples for all items with different content
    """
    difference = []

    for i in a.identifiers:
        fpath = a.item_content_abspath(i)
        calc_hash = reference._storage_broker.hasher(fpath)
        ref_hash = reference.item_properties(i)["hash"]
        if calc_hash != ref_hash:
            info = (i, calc_hash, ref_hash)
            difference.append(info)
        if progressbar:
            progressbar.update(1)

    return difference