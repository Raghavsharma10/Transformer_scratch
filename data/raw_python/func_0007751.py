def flatten_multi_dim(sequence):
    """Flatten a multi-dimensional array-like to a single dimensional sequence
    (as a generator).
    """
    for x in sequence:
        if (isinstance(x, collections.Iterable)
                and not isinstance(x, six.string_types)):
            for y in flatten_multi_dim(x):
                yield y
        else:
            yield x