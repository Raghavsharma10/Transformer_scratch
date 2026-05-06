def mad(data, axis=None):
    """
    Computes the median absolute deviation of *data* along a given *axis*.
    See `link <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_ for
    details.

    **Parameters**

    data : array-like

    **Returns**

    mad : number or array-like
    """
    return median(absolute(data - median(data, axis)), axis)