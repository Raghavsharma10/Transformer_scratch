def hist_last_nonzero(hist, return_index=False, copy=True):
    '''Find the last nonzero index and mask the remaining entries.

    Parameters
    ----------
    hist : array_like, iterable
        Input histogram with dimension at most 1.
    return_index : bool, optional
        If true, return the index.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead. Default is False.

    Returns
    -------
    masked_hist : masked_array
       Hist with masked elements.
    masked_hist : masked_array, tuple
        Hist with masked elements and index of the element after the last nonzero value.
    '''
    # make np array
    hist_t = np.array(hist)
    index = (np.where(hist_t)[-1][-1] + 1) if np.sum(hist_t) > 1 else hist_t.shape[0]
    # copy and create ma
    masked_hist = np.ma.array(hist, copy=copy, mask=True)
    masked_hist.mask[index:] = False
    if return_index:
        return masked_hist, index
    else:
        return masked_hist