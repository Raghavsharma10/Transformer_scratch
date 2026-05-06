def _cdist(x, y, exponent=1):
    """
    Pairwise distance between points in two sets.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_double(x) and _can_be_double(y):
        return _cdist_scipy(x, y, exponent)
    else:
        return _cdist_naive(x, y, exponent)