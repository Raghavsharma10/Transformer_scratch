def random_like(ary=None, shape=None, dtype=None):
    """
    Returns a random array of the same shape and type as the
    supplied array argument, or the supplied shape and dtype
    """
    if ary is not None:
        shape, dtype = ary.shape, ary.dtype
    elif shape is None or dtype is None:
        raise ValueError((
            'random_like(ary, shape, dtype) must be supplied '
            'with either an array argument, or the shape and dtype '
            'of the desired random array.'))

    if np.issubdtype(dtype, np.complexfloating):
        return (np.random.random(size=shape) + \
            np.random.random(size=shape)*1j).astype(dtype)
    else:
        return np.random.random(size=shape).astype(dtype)