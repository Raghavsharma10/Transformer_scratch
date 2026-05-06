def as_vec4(obj, default=(0, 0, 0, 1)):
    """
    Convert `obj` to 4-element vector (numpy array with shape[-1] == 4)

    Parameters
    ----------
    obj : array-like
        Original object.
    default : array-like
        The defaults to use if the object does not have 4 entries.

    Returns
    -------
    obj : array-like
        The object promoted to have 4 elements.

    Notes
    -----
    `obj` will have at least two dimensions.

    If `obj` has < 4 elements, then new elements are added from `default`.
    For inputs intended as a position or translation, use default=(0,0,0,1).
    For inputs intended as scale factors, use default=(1,1,1,1).

    """
    obj = np.atleast_2d(obj)
    # For multiple vectors, reshape to (..., 4)
    if obj.shape[-1] < 4:
        new = np.empty(obj.shape[:-1] + (4,), dtype=obj.dtype)
        new[:] = default
        new[..., :obj.shape[-1]] = obj
        obj = new
    elif obj.shape[-1] > 4:
        raise TypeError("Array shape %s cannot be converted to vec4"
                        % obj.shape)
    return obj