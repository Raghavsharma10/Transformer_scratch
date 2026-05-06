def nTimes(n, f, *args, **kwargs):
    r"""Call `f` `n` times with `args` and `kwargs`.
    Useful e.g. for simplistic timing.

    Examples:

    >>> nTimes(3, sys.stdout.write, 'hallo\n')
    hallo
    hallo
    hallo

    """
    for i in xrange(n): f(*args, **kwargs)