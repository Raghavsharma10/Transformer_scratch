def _sliced_shape(shape, keys):
    """
    Returns the shape that results from slicing an array of the given
    shape by the given keys.

    >>> _sliced_shape(shape=(52350, 70, 90, 180),
    ...               keys=(np.newaxis, slice(None, 10), 3,
    ...                     slice(None), slice(2, 3)))
    (1, 10, 90, 1)

    """
    keys = _full_keys(keys, len(shape))

    sliced_shape = []
    shape_dim = -1
    for key in keys:
        shape_dim += 1
        if _is_scalar(key):
            continue
        elif isinstance(key, slice):
            size = len(range(*key.indices(shape[shape_dim])))
            sliced_shape.append(size)
        elif isinstance(key, np.ndarray) and key.dtype == np.dtype('bool'):
            # Numpy boolean indexing.
            sliced_shape.append(builtins.sum(key))
        elif isinstance(key, (tuple, np.ndarray)):
            sliced_shape.append(len(key))
        elif key is np.newaxis:
            shape_dim -= 1
            sliced_shape.append(1)
        else:
            raise ValueError('Invalid indexing object "{}"'.format(key))

    sliced_shape = tuple(sliced_shape)
    return sliced_shape