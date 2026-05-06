def NonNegativeInt(n):
    """If *n* is non-negative integer returns it, otherwise an error.

    >>> print("%d" % NonNegativeInt('8'))
    8

    >>> NonNegativeInt('8.1')
    Traceback (most recent call last):
       ...
    ValueError: 8.1 is not an integer

    >>> print("%d" % NonNegativeInt('0'))
    0

    >>> NonNegativeInt('-1')
    Traceback (most recent call last):
       ...
    ValueError: -1 is not non-negative

    """
    if not isinstance(n, str):
        raise ValueError('%r is not a string' % n)
    try:
       n = int(n)
    except:
        raise ValueError('%s is not an integer' % n)
    if n < 0:
        raise ValueError('%d is not non-negative' % n)
    else:
        return n