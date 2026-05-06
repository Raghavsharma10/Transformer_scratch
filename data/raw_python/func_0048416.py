def all_equal(iterable):
    """Checks whether all items in an iterable are equal.

    Parameters
    ----------
    iterable
        An iterable, e.g. a string og a list.

    Returns
    -------
    boolean
        True or False.
    
    Examples
    -------
    >>> all_equal([2, 2, 2])
    True
    >>> all_equal([1, 2, 3])
    False
    """
    if len(iterable) in [0, 1]:
        return False

    first = iterable[0]
    return all([first == i for i in iterable[1:]])