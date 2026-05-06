def funcSmthTmp(aryFuncChnk, varSdSmthTmp):
    """Apply temporal smoothing to fMRI data & pRF time course models.

    Parameters
    ----------
    aryFuncChnk : np.array
        TODO
    varSdSmthTmp : float (?)
        extend of smoothing

    Returns
    -------
    aryFuncChnk : np.array
        TODO
    """
    # For the filtering to perform well at the ends of the time series, we
    # set the method to 'nearest' and place a volume with mean intensity
    # (over time) at the beginning and at the end.
    aryFuncChnkMean = np.mean(aryFuncChnk,
                              axis=1,
                              keepdims=True)

    aryFuncChnk = np.concatenate((aryFuncChnkMean,
                                  aryFuncChnk,
                                  aryFuncChnkMean), axis=1)

    # In the input data, time goes from left to right. Therefore, we apply
    # the filter along axis=1.
    aryFuncChnk = gaussian_filter1d(aryFuncChnk,
                                    varSdSmthTmp,
                                    axis=1,
                                    order=0,
                                    mode='nearest',
                                    truncate=4.0)

    # Remove mean-intensity volumes at the beginning and at the end:
    aryFuncChnk = aryFuncChnk[:, 1:-1]

    # Output list:
    return aryFuncChnk