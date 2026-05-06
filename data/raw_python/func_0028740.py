def mean(a, axis=None, mdtol=1):
    """
    Request the mean of an Array over any number of axes.

    .. note:: Currently limited to operating on a single axis.

    :param axis: Axis or axes along which the operation is performed.
                 The default (axis=None) is to perform the operation
                 over all the dimensions of the input array.
                 The axis may be negative, in which case it counts from
                 the last to the first axis.
                 If axis is a tuple of ints, the operation is performed
                 over multiple axes.
    :type axis: None, or int, or iterable of ints.
    :param float mdtol: Tolerance of missing data. The value in each
                        element of the resulting array will be masked if the
                        fraction of masked data contributing to that element
                        exceeds mdtol. mdtol=0 means no missing data is
                        tolerated while mdtol=1 will mean the resulting
                        element will be masked if and only if all the
                        contributing elements of the source array are masked.
                        Defaults to 1.
    :return: The Array representing the requested mean.
    :rtype: Array

    """
    axes = _normalise_axis(axis, a)
    if axes is None or len(axes) != 1:
        msg = "This operation is currently limited to a single axis"
        raise AxisSupportError(msg)
    dtype = (np.array([0], dtype=a.dtype) / 1.).dtype
    kwargs = dict(mdtol=mdtol)
    return _Aggregation(a, axes[0],
                        _MeanStreamsHandler, _MeanMaskedStreamsHandler,
                        dtype, kwargs)