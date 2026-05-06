def loader(pattern, dimensions=None, distributed_dim='time', read_only=False):
    """
    It provide a root descriptor to be used inside a with statement. It
    automatically close the root when the with statement finish.

    Keyword arguments:
    root -- the root descriptor returned by the 'open' function
    """
    if dimensions:
        root = tailor(pattern, dimensions, distributed_dim, read_only=read_only)
    else:
        root, _ = open(pattern, read_only=read_only)
    yield root
    root.close()