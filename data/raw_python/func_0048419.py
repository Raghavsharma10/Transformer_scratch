def spread(iterable):
    """Returns the maximal spread of a sorted list of numbers.

    Parameters
    ----------
    iterable
        A list of numbers.

    Returns
    -------
    max_diff
        The maximal difference when the iterable is sorted.


    Examples
    -------
    >>> spread([1, 11, 13, 15])
    10
    
    >>> spread([1, 15, 11, 13])
    10
    """
    if len(iterable) == 1:
        return 0

    iterable = iterable.copy()
    iterable.sort()

    max_diff = max(abs(i - j) for (i, j) in zip(iterable[1:], iterable[:-1]))

    return max_diff