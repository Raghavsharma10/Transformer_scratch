def _mode(elems):
    """
    Find the mode (most common element) in list elems. If there are ties, this function returns the least value.

    If elems is an empty list, returns None.
    """
    if len(elems) == 0:
        return None

    c = collections.Counter()
    c.update(elems)

    most_common = c.most_common(1)
    most_common.sort()
    return most_common[0][0]