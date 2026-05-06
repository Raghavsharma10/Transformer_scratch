def funcSmthSpt(aryFuncChnk, varSdSmthSpt):
    """Apply spatial smoothing to the input data.

    Parameters
    ----------
    aryFuncChnk : np.array
        TODO
    varSdSmthSpt : float (?)
        Extent of smoothing.
    Returns
    -------
    aryFuncChnk : np.array
        Smoothed data.
    """
    varNdim = aryFuncChnk.ndim

    # Number of time points in this chunk:
    varNumVol = aryFuncChnk.shape[-1]

    # Loop through volumes:
    if varNdim == 4:
        for idxVol in range(0, varNumVol):

            aryFuncChnk[:, :, :, idxVol] = gaussian_filter(
                aryFuncChnk[:, :, :, idxVol],
                varSdSmthSpt,
                order=0,
                mode='nearest',
                truncate=4.0)
    elif varNdim == 5:
        varNumMtnDrctns = aryFuncChnk.shape[3]
        for idxVol in range(0, varNumVol):
            for idxMtn in range(0, varNumMtnDrctns):
                aryFuncChnk[:, :, :, idxMtn, idxVol] = gaussian_filter(
                    aryFuncChnk[:, :, :, idxMtn, idxVol],
                    varSdSmthSpt,
                    order=0,
                    mode='nearest',
                    truncate=4.0)

    # Output list:
    return aryFuncChnk