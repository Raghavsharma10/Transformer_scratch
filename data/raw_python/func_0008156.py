def confidence_interval_arr(data, conf=0.95):
    r""" Computes element-wise confidence intervals from a sample of ndarrays

    Given a sample of arbitrarily shaped ndarrays, computes element-wise confidence intervals

    Parameters
    ----------
    data : ndarray (K, (shape))
        ndarray of ndarrays, the first index is a sample index, the remaining indexes are specific to the
        array of interest
    conf : float, optional, default = 0.95
        confidence interval

    Return
    ------
    lower : ndarray(shape)
        element-wise lower bounds
    upper : ndarray(shape)
        element-wise upper bounds

    """
    if conf < 0 or conf > 1:
        raise ValueError('Not a meaningful confidence level: '+str(conf))

    # list or 1D-array? then fuse it
    if types.is_list(data) or (isinstance(data, np.ndarray) and np.ndim(data) == 1):
        newshape = tuple([len(data)] + list(data[0].shape))
        newdata = np.zeros(newshape)
        for i in range(len(data)):
            newdata[i, :] = data[i]
        data = newdata

    # do we have an array now? if yes go, if no fail
    if types.is_float_array(data):
        I = _indexes(data[0])
        lower = np.zeros(data[0].shape)
        upper = np.zeros(data[0].shape)
        for i in I:
            col = _column(data, i)
            m, lower[i], upper[i] = confidence_interval(col, conf)
        # return
        return lower, upper
    else:
        raise TypeError('data cannot be converted to an ndarray')