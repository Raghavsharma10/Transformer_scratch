def cnvlGauss2D(idxPrc, aryBoxCar, aryMdlParamsChnk, tplPngSize, varNumVol,
                queOut):
    """Spatially convolve boxcar functions with 2D Gaussian.

    Parameters
    ----------
    idxPrc : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    aryBoxCar : float, positive
      Description of input 2.
    aryMdlParamsChnk : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    tplPngSize : float, positive
      Description of input 2.
    varNumVol : 2d numpy array, shape [n_samples, n_measurements]
        Description of input 1.
    queOut : float, positive
      Description of input 2.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Determine number of motion directions
    varNumMtnDrtn = aryBoxCar.shape[2]

    # Output array with pRF model time courses:
    aryOut = np.zeros([varChnkSze, varNumMtnDrtn, varNumVol])

    # Loop through different motion directions:
    for idxMtn in range(0, varNumMtnDrtn):
        # Loop through combinations of model parameters:
        for idxMdl in range(0, varChnkSze):

            # Spatial parameters of current model:
            varTmpX = aryMdlParamsChnk[idxMdl, 1]
            varTmpY = aryMdlParamsChnk[idxMdl, 2]
            varTmpSd = aryMdlParamsChnk[idxMdl, 3]

            # Create pRF model (2D):
            aryGauss = crtGauss2D(tplPngSize[0],
                                  tplPngSize[1],
                                  varTmpX,
                                  varTmpY,
                                  varTmpSd)

            # Multiply pixel-time courses with Gaussian pRF models:
            aryPrfTcTmp = np.multiply(aryBoxCar[:, :, idxMtn, :],
                                      aryGauss[:, :, None])

            # Calculate sum across x- and y-dimensions - the 'area under the
            # Gaussian surface'. This is essentially an unscaled version of the
            # pRF time course model (i.e. not yet scaled for size of the pRF).
            aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1))

            # Put model time courses into function's output with 2d Gaussian
            # arrray:
            aryOut[idxMdl, idxMtn, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    lstOut = [idxPrc,
              aryOut]

    # Put output to queue:
    queOut.put(lstOut)