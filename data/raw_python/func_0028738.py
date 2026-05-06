def max(a, axis=None):
    """
    Request the maximum of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    Parameters
    ----------
    a : Array object
        The object whose maximum is to be found.
    axis : None, or int, or iterable of ints
        Axis or axes along which the operation is performed. The default
        (axis=None) is to perform the operation over all the dimensions of the
        input array. The axis may be negative, in which case it counts from
        the last to the first axis. If axis is a tuple of ints, the operation
        is performed over multiple axes.

    Returns
    -------
    out : Array
        The Array representing the requested max.
    """
    axes = _normalise_axis(axis, a)
    assert axes is not None and len(axes) == 1
    return _Aggregation(a, axes[0],
                        _MaxStreamsHandler, _MaxMaskedStreamsHandler,
                        a.dtype, {})