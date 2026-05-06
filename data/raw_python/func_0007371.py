def iterate_chunks(i, size=10):
    """
    Iterate over an iterator ``i`` in ``size`` chunks, yield chunks.
    Similar to pagination.

    Example::

        >>> list(iterate_chunks([1, 2, 3, 4], size=2))
        [[1, 2], [3, 4]]
    """
    accumulator = []

    for n, i in enumerate(i):
        accumulator.append(i)
        if (n+1) % size == 0:
            yield accumulator
            accumulator = []

    if accumulator:
        yield accumulator