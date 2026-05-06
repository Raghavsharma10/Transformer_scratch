def hist_quantiles(hist, prob=(0.05, 0.95), return_indices=False, copy=True):
    '''Calculate quantiles from histograms, cuts off hist below and above given quantile. This function will not cut off more than the given values.

    Parameters
    ----------
    hist : array_like, iterable
        Input histogram with dimension at most 1.
    prob : float, list, tuple
        List of quantiles to compute. Upper and lower limit. From 0 to 1. Default is 0.05 and 0.95.
    return_indices : bool, optional
        If true, return the indices of the hist.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead. Default is False.

    Returns
    -------
    masked_hist : masked_array
       Hist with masked elements.
    masked_hist : masked_array, tuple
        Hist with masked elements and indices.
    '''
    # make np array
    hist_t = np.array(hist)
    # calculate cumulative distribution
    cdf = np.cumsum(hist_t)
    # copy, convert and normalize
    if cdf[-1] == 0:
        normcdf = cdf.astype('float')
    else:
        normcdf = cdf.astype('float') / cdf[-1]
    # calculate unique values from cumulative distribution and their indices
    unormcdf, indices = np.unique(normcdf, return_index=True)
    # calculate limits
    try:
        hp = np.where(unormcdf > prob[1])[0][0]
        lp = np.where(unormcdf >= prob[0])[0][0]
    except IndexError:
        hp_index = hist_t.shape[0]
        lp_index = 0
    else:
        hp_index = indices[hp]
        lp_index = indices[lp]
    # copy and create ma
    masked_hist = np.ma.array(hist, copy=copy, mask=True)
    masked_hist.mask[lp_index:hp_index + 1] = False
    if return_indices:
        return masked_hist, (lp_index, hp_index)
    else:
        return masked_hist