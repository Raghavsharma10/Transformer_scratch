def percentile(values, percent):
    """
    PERCENTILE WITH INTERPOLATION
    RETURN VALUE AT, OR ABOVE, percentile OF THE VALUES

    snagged from http://code.activestate.com/recipes/511478-finding-the-percentile-of-the-values/
    """
    N = sorted(values)
    if not N:
        return None
    k = (len(N) - 1) * percent
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return N[int(k)]
    d0 = N[f] * (c - k)
    d1 = N[c] * (k - f)
    return d0 + d1