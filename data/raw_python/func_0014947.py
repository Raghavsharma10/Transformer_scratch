def argmin(iterable, key=None, both=False):
    """See `argmax`.
    """
    if key is not None:
        it = imap(key, iterable)
    else:
        it = iter(iterable)
    score, argmin = reduce(min, izip(it, count()))
    if both:
        return argmin, score
    return argmin