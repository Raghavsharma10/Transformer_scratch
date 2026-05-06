def FloatGreaterThanEqualToZero(x):
    """If *x* is a float >= 0, returns it, otherwise raises and error.

    >>> print('%.1f' % FloatGreaterThanEqualToZero('1.5'))
    1.5

    >>> print('%.1f' % FloatGreaterThanEqualToZero('-1.1'))
    Traceback (most recent call last):
       ...
    ValueError: -1.1 not float greater than or equal to zero
    """
    try:
        x = float(x)
    except:
        raise ValueError("%r not float greater than or equal to zero" % x)
    if x >= 0:
        return x
    else:
        raise ValueError("%r not float greater than or equal to zero" % x)