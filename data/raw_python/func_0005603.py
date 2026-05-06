def pstdev(data):
    """Calculates the population standard deviation."""
    #: http://stackoverflow.com/a/27758326
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n  # the population variance
    return pvar**0.5