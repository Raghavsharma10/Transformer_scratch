def tailor(pattern_or_root, dimensions=None, distributed_dim='time',
           read_only=False):
    """
    Return a TileManager to wrap the root descriptor and tailor all the
    dimensions to a specified window.

    Keyword arguments:
    root -- a NCObject descriptor.
    pattern -- a filename string to open a NCObject descriptor.
    dimensions -- a dictionary to configurate the dimensions limits.
    """
    return TileManager(pattern_or_root, dimensions=dimensions,
                       distributed_dim=distributed_dim, read_only=read_only)