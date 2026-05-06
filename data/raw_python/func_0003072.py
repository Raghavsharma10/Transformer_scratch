def prep_models(aryPrfTc, varSdSmthTmp=2.0, lgcPrint=True):
    """
    Prepare pRF model time courses.

    Parameters
    ----------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    varSdSmthTmp : float
        Extent of temporal smoothing that is applied to functional data and
        pRF time course models, [SD of Gaussian kernel, in seconds]. If `zero`,
        no temporal smoothing is applied.
    lgcPrint : boolean
        Whether print statements should be executed.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with prepared pRF time course models, same
        dimensions as input (`aryPrfTc[x-position, y-position, SD, volume]`).
    """
    if lgcPrint:
        print('------Prepare pRF time course models')

    # Define temporal smoothing of pRF time course models
    def funcSmthTmp(aryPrfTc, varSdSmthTmp, lgcPrint=True):
        """Apply temporal smoothing to fMRI data & pRF time course models.

        Parameters
        ----------
        aryPrfTc : np.array
            4D numpy array with pRF time course models, with following
            dimensions: `aryPrfTc[x-position, y-position, SD, volume]`.
        varSdSmthTmp : float, positive
            Extent of temporal smoothing that is applied to functional data and
            pRF time course models, [SD of Gaussian kernel, in seconds]. If
            `zero`, no temporal smoothing is applied.
        lgcPrint : boolean
            Whether print statements should be executed.

        Returns
        -------
        aryPrfTc : np.array
            4D numpy array with prepared pRF time course models, same dimension
            as input (`aryPrfTc[x-position, y-position, SD, volume]`).
        """

        # adjust the input, if necessary, such that input is 2D, with last
        # dim time
        tplInpShp = aryPrfTc.shape
        aryPrfTc = aryPrfTc.reshape((-1, aryPrfTc.shape[-1]))

        # For the filtering to perform well at the ends of the time series, we
        # set the method to 'nearest' and place a volume with mean intensity
        # (over time) at the beginning and at the end.
        aryPrfTcMean = np.mean(aryPrfTc, axis=-1, keepdims=True).reshape(-1, 1)

        aryPrfTc = np.concatenate((aryPrfTcMean, aryPrfTc, aryPrfTcMean),
                                  axis=-1)

        # In the input data, time goes from left to right. Therefore, we apply
        # the filter along axis=1.
        aryPrfTc = gaussian_filter1d(aryPrfTc.astype('float32'), varSdSmthTmp,
                                     axis=-1, order=0, mode='nearest',
                                     truncate=4.0)

        # Remove mean-intensity volumes at the beginning and at the end:
        aryPrfTc = aryPrfTc[..., 1:-1]

        # Output array:
        return aryPrfTc.reshape(tplInpShp).astype('float16')

    # Perform temporal smoothing of pRF time course models
    if 0.0 < varSdSmthTmp:
        if lgcPrint:
            print('---------Temporal smoothing on pRF time course models')
            print('------------SD tmp smooth is: ' + str(varSdSmthTmp))
        aryPrfTc = funcSmthTmp(aryPrfTc, varSdSmthTmp)

    # Z-score the prf time course models
    if lgcPrint:
        print('---------Zscore the pRF time course models')
    # De-mean the prf time course models:
    aryPrfTc = np.subtract(aryPrfTc, np.mean(aryPrfTc, axis=-1)[..., None])

    # Standardize the prf time course models:
    # In order to avoid devision by zero, only divide those voxels with a
    # standard deviation greater than zero:
    aryTmpStd = np.std(aryPrfTc, axis=-1)
    aryTmpLgc = np.greater(aryTmpStd, np.array([0.0]))
    aryPrfTc[aryTmpLgc, :] = np.divide(aryPrfTc[aryTmpLgc, :],
                                       aryTmpStd[aryTmpLgc, None])

    return aryPrfTc