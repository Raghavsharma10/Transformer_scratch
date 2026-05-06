def var(a, axis=None, ddof=0):
    """
    Request the variance of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :param int ddof: Delta Degrees of Freedom. The divisor used in
                     calculations is N - ddof, where N represents the
                     number of elements. By default ddof is zero.
    :return: The Array representing the requested variance.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    dtype = (np.array([0], dtype=a.dtype) / 1.).dtype
    return _Aggregation(a, axes[0],
                        _VarStreamsHandler, _VarMaskedStreamsHandler,
                        dtype, dict(ddof=ddof))