def argmax(iterable, key=None, both=False):
    """
    >>> argmax([4,2,-5])
    0
    >>> argmax([4,2,-5], key=abs)
    2
    >>> argmax([4,2,-5], key=abs, both=True)
    (2, 5)
    """
    if key is not None:
        it = imap(key, iterable)
    else:
        it = iter(iterable)
    score, argmax = reduce(max, izip(it, count()))
    if both:
        return argmax, score
    return argmax