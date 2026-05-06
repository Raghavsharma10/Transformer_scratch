def crtPwBoxCarFn(varNumVol, aryPngData, aryPresOrd, vecMtDrctn):
    """Create pixel-wise boxcar functions.

    Parameters
    ----------
    input1 : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    input2 : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Create pixel-wise boxcar functions')
    aryBoxCar = np.empty(aryPngData.shape[0:2] + (len(vecMtDrctn),) +
                         (varNumVol,), dtype='int64')
    for ind, num in enumerate(vecMtDrctn):
        aryCondTemp = np.zeros((aryPngData.shape), dtype='int64')
        lgcTempMtDrctn = [aryPresOrd == num][0]
        aryCondTemp[:, :, lgcTempMtDrctn] = np.copy(
            aryPngData[:, :, lgcTempMtDrctn])
        aryBoxCar[:, :, ind, :] = aryCondTemp

    return aryBoxCar