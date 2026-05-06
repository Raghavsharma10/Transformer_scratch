def _flatten(n):
    """Recursively flatten a mixed sequence of sub-sequences and items"""
    if isinstance(n, collections.Sequence):
        for x in n:
            for y in _flatten(x):
                yield y
    else:
        yield n