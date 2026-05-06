def crt_nrl_tc(aryMdlRsp, aryCnd, aryOns, aryDrt, varTr, varNumVol,
               varTmpOvsmpl, lgcPrint=True):
    """Create temporally upsampled neural time courses.

    Parameters
    ----------
    aryMdlRsp : 2d numpy array, shape [n_x_pos * n_y_pos * n_sd, n_cond]
        Responses of 2D Gauss models to spatial conditions.
    aryCnd : np.array
        1D array with condition identifiers (every condition has its own int)
    aryOns : np.array, same len as aryCnd
        1D array with condition onset times in seconds.
    aryDrt : np.array, same len as aryCnd
        1D array with condition durations of different conditions in seconds.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment
    varNumVol : float, positive
        Number of data point (volumes) in the (fMRI) data
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.
    lgcPrint: boolean, default True
        Should print messages be sent to user?

    Returns
    -------
    aryNrlTc : 2d numpy array,
               shape [n_x_pos * n_y_pos * n_sd, varNumVol*varTmpOvsmpl]
        Neural time course models in temporally upsampled space

    Notes
    ---------
    [1] This function first creates boxcar functions based on the  conditions
        as they are specified in the temporal experiment information, provided
        by the user in the csv file. Second, it then replaces the 1s in the
        boxcar function by predicted condition values that were previously
        calculated based on the overlap between the assumed 2D Gaussian for the
        current model and the presented stimulus aperture for that condition.
        Since the 2D Gaussian is normalized, the overlap value will be between
        0 and 1.

    """

    # adjust the input, if necessary, such that input is 2D
    tplInpShp = aryMdlRsp.shape
    aryMdlRsp = aryMdlRsp.reshape((-1, aryMdlRsp.shape[-1]))

    # the first spatial condition might code the baseline (blank periods) with
    # all zeros. If this is the case, remove the first spatial condition, since
    # for temporal conditions this is removed automatically below and we need
    # temporal and sptial conditions to maych
    if np.all(aryMdlRsp[:, 0] == 0):
        if lgcPrint:
            print('------------Removed first spatial condition (all zeros)')
        aryMdlRsp = aryMdlRsp[:, 1:]

    # create boxcar functions in temporally upsampled space
    aryBxCarTmp = create_boxcar(aryCnd, aryOns, aryDrt, varTr, varNumVol,
                                aryExclCnd=np.array([0.]),
                                varTmpOvsmpl=varTmpOvsmpl).T

    # Make sure that aryMdlRsp and aryBxCarTmp have the same number of
    # conditions
    assert aryMdlRsp.shape[-1] == aryBxCarTmp.shape[0]

    # pre-allocate pixelwise boxcar array
    aryNrlTc = np.zeros((aryMdlRsp.shape[0], aryBxCarTmp.shape[-1]),
                        dtype='float16')
    # loop through boxcar functions of conditions
    for ind, vecCndOcc in enumerate(aryBxCarTmp):
        # get response predicted by models for this specific spatial condition
        rspValPrdByMdl = aryMdlRsp[:, ind]
        # insert predicted response value several times using broad-casting
        aryNrlTc[..., vecCndOcc.astype('bool')] = rspValPrdByMdl[:, None]

    # determine output shape
    tplOutShp = tplInpShp[:-1] + (int(varNumVol*varTmpOvsmpl), )

    return aryNrlTc.reshape(tplOutShp).astype('float16')