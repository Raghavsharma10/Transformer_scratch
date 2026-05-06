def _pdist(x, exponent=1):
    """
    Pairwise distance between points in a set.

    As Scipy converts every value to double, this wrapper uses
    a less efficient implementation if the original dtype
    can not be converted to double.

    """
    if _can_be_double(x):
        return _pdist_scipy(x, exponent)
    else:
        return _cdist_naive(x, x, exponent)