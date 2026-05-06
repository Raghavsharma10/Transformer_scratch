def bisect_index(a, x):
    """ Find the leftmost index of an element in a list using binary search.

    Parameters
    ----------
    a: list
        A sorted list.
    x: arbitrary
        The element.

    Returns
    -------
    int
        The index.

    """
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError