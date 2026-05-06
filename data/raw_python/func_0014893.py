def div(a,b):
    """``div(a,b)`` is like ``a // b`` if ``b`` devides ``a``, otherwise
    an `ValueError` is raised.

    >>> div(10,2)
    5
    >>> div(10,3)
    Traceback (most recent call last):
    ...
    ValueError: 3 does not divide 10
    """
    res, fail = divmod(a,b)
    if fail:
        raise ValueError("%r does not divide %r" % (b,a))
    else:
        return res