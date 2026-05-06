def ibatch(iterable, size):
    """Yield a series of batches from iterable, each size elements long."""
    source = iter(iterable)
    while True:
        batch = itertools.islice(source, size)
        yield itertools.chain([next(batch)], batch)