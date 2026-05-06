def flatten_iterable(iterable):
    """flatten iterable, but leaves out strings

    [[[1, 2, 3], [4, 5]], 6] -> [1, 2, 3, 4, 5, 6]

    """
    for item in iterable:
        if isinstance(item, collections.Iterable) and not isinstance(item, basestring):
            for sub in flatten_iterable(item):
                yield sub
        else:
            yield item