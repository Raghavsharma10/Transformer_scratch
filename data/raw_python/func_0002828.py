def crt_prf_ftr_tc(aryMdlRsp, aryTmpExpInf, varNumVol, varTr, varTmpOvsmpl,
                   switchHrfSet, tplPngSize, varPar, dctPrm=None,
                   lgcPrint=True):
    """Create all spatial x feature prf time courses.

    Parameters
    ----------
    aryMdlRsp : 2d numpy array, shape [n_x_pos * n_y_pos * n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions
    aryTmpExpInf: 2d numpy array, shape [unknown, 4]
        Temporal information about conditions
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varTmpOvsmpl : int, positive
        Factor by which the data hs been temporally upsampled.
    switchHrfSet : int, (1, 2, 3)
        Switch to determine which hrf basis functions are used
    tplPngSize : tuple
        Pixel dimensions of the visual space (width, height).
    varPar : int, positive
        Description of input 1.
    dctPrm : dictionary, default None
        Dictionary with customized hrf parameters. If this is None, default
        hrf parameters will be used.
    lgcPrint: boolean, default True
        Should print messages be sent to user?

    Returns
    -------
    aryNrlTcConv : 3d numpy array,
                   shape [nr of models, nr of unique feautures, varNumVol]
        Prf time course models

    """

    # Identify number of unique features
    vecFeat = np.unique(aryTmpExpInf[:, 3])
    vecFeat = vecFeat[np.nonzero(vecFeat)[0]]

    # Preallocate the output array
    aryPrfTc = np.zeros((aryMdlRsp.shape[0], 0, varNumVol),
                        dtype=np.float32)

    # Loop over unique features
    for indFtr, ftr in enumerate(vecFeat):

        if lgcPrint:
            print('---------Create prf time course model for feature ' +
                  str(ftr))
        # Derive sptial conditions, onsets and durations for this specific
        # feature
        aryTmpCnd = aryTmpExpInf[aryTmpExpInf[:, 3] == ftr, 0]
        aryTmpOns = aryTmpExpInf[aryTmpExpInf[:, 3] == ftr, 1]
        aryTmpDrt = aryTmpExpInf[aryTmpExpInf[:, 3] == ftr, 2]

        # Create temporally upsampled neural time courses.
        aryNrlTcTmp = crt_nrl_tc(aryMdlRsp, aryTmpCnd, aryTmpOns, aryTmpDrt,
                                 varTr, varNumVol, varTmpOvsmpl,
                                 lgcPrint=lgcPrint)
        # Convolve with hrf to create model pRF time courses.
        aryPrfTcTmp = crt_prf_tc(aryNrlTcTmp, varNumVol, varTr, varTmpOvsmpl,
                                 switchHrfSet, tplPngSize, varPar,
                                 dctPrm=dctPrm, lgcPrint=lgcPrint)
        # Add temporal time course to time course that will be returned
        aryPrfTc = np.concatenate((aryPrfTc, aryPrfTcTmp), axis=1)

    return aryPrfTc