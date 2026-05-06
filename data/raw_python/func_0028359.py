def chunks(iterable, chunksize, cast=tuple):
    # type: (Iterable, int, Callable) -> Iterable
    """
        Yields items from an iterator in iterable chunks.
    """
    it = iter(iterable)
    while True:
        yield cast(itertools.chain([next(it)],
                   itertools.islice(it, chunksize - 1)))