def _chunks(iterable, n):
    """
    Splits an iterable into chunks of size n.
    """
    iterable = iter(iterable)
    while True:
        # store one line in memory,
        # chain it to an iterator on the rest of the chunk
        yield chain([next(iterable)], islice(iterable, n-1))