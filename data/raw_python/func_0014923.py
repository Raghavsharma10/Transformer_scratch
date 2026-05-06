def invertDict(d, allowManyToOne=False):
    r"""Return an inverted version of dict `d`, so that values become keys and
    vice versa. If multiple keys in `d` have the same value an error is
    raised, unless `allowManyToOne` is true, in which case one of those
    key-value pairs is chosen at random for the inversion.

    Examples:

    >>> invertDict({1: 2, 3: 4}) == {2: 1, 4: 3}
    True
    >>> invertDict({1: 2, 3: 2})
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    ValueError: d can't be inverted!
    >>> invertDict({1: 2, 3: 2}, allowManyToOne=True).keys()
    [2]
    """
    res = dict(izip(d.itervalues(), d.iterkeys()))
    if not allowManyToOne and len(res) != len(d):
        raise ValueError("d can't be inverted!")
    return res