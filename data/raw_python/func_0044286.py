def worker(n):
    """Spend some time calculating exponentials."""
    for _ in xrange(999999):
        a = exp(n)
        b = exp(2*n)
    return n, a