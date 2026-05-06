def mapConcat(func, *iterables):
    """Similar to `map` but the instead of collecting the return values of
    `func` in a list, the items of each return value are instaed collected
    (so `func` must return an iterable type).

    Examples:

    >>> mapConcat(lambda x:[x], [1,2,3])
    [1, 2, 3]
    >>> mapConcat(lambda x: [x,str(x)], [1,2,3])
    [1, '1', 2, '2', 3, '3']
    """
    return [e for l in imap(func, *iterables) for e in l]