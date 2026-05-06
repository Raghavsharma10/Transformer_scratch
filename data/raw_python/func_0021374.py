def validate_inputs(*arrays, **kwargs):
    """Validate input arrays

    This checks that
    - Arrays are mutually broadcastable
    - Broadcasted arrays are one-dimensional

    Optionally, arrays are sorted according to the ``sort_by`` argument.

    Parameters
    ----------
    *args : ndarrays
        All non-keyword arguments are arrays which will be validated
    sort_by : array
        If specified, sort all inputs by the order given in this array.
    """
    arrays = np.broadcast_arrays(*arrays)
    sort_by = kwargs.pop('sort_by', None)

    if kwargs:
        raise ValueError("unrecognized arguments: {0}".format(kwargs.keys()))

    if arrays[0].ndim != 1:
        raise ValueError("Input arrays should be one-dimensional.")

    if sort_by is not None:
        isort = np.argsort(sort_by)
        if isort.shape != arrays[0].shape:
            raise ValueError("sort shape must equal array shape.")
        arrays = tuple([a[isort] for a in arrays])
    return arrays