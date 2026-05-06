def cnvl_2D_gauss(idxPrc, aryMdlParamsChnk, arySptExpInf, tplPngSize, queOut,
                  strCrd='crt'):
    """Spatially convolve input with 2D Gaussian model.

    Parameters
    ----------
    idxPrc : int
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryMdlParamsChnk : 2d numpy array, shape [n_models, n_model_params]
        Array with the model parameter combinations for this chunk.
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2.
        Pixel dimensions of the visual space (width, height).
    queOut : multiprocessing.queues.Queue
        Queue to put the results on. If this is None, the user is not running
        multiprocessing but is just calling the function
    strCrd, string, either 'crt' or 'pol'
        Whether model parameters are provided in cartesian or polar coordinates

    Returns
    -------
    data : 2d numpy array, shape [n_models, n_conditions]
        Closed data.
    Reference
    ---------
    [1]
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = aryMdlParamsChnk.shape[0]

    # Number of conditions / time points of the input data
    varNumLstAx = arySptExpInf.shape[-1]

    # Output array with results of convolution:
    aryOut = np.zeros((varChnkSze, varNumLstAx))

    # Loop through combinations of model parameters:
    for idxMdl in range(0, varChnkSze):

        # Spatial parameters of current model:
        if strCrd == 'pol':
            # Position was given in polar coordinates
            varTmpEcc = aryMdlParamsChnk[idxMdl, 0]
            varTmpPlrAng = aryMdlParamsChnk[idxMdl, 1]
            # Convert from polar to to cartesian coordinates
            varTmpX = varTmpEcc * np.cos(varTmpPlrAng) + tplPngSize[0]/2.
            varTmpY = varTmpEcc * np.sin(varTmpPlrAng) + tplPngSize[1]/2.

        elif strCrd == 'crt':
            varTmpX = aryMdlParamsChnk[idxMdl, 0]
            varTmpY = aryMdlParamsChnk[idxMdl, 1]

        # Standard deviation does not depend on coordinate system
        varTmpSd = aryMdlParamsChnk[idxMdl, 2]

        # Create pRF model (2D):
        aryGauss = crt_2D_gauss(tplPngSize[0],
                                tplPngSize[1],
                                varTmpX,
                                varTmpY,
                                varTmpSd)

        # Multiply pixel-time courses with Gaussian pRF models:
        aryCndTcTmp = np.multiply(arySptExpInf, aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'.
        aryCndTcTmp = np.sum(aryCndTcTmp, axis=(0, 1))

        # Put model time courses into function's output with 2d Gaussian
        # arrray:
        aryOut[idxMdl, :] = aryCndTcTmp

    if queOut is None:
        # if user is not using multiprocessing, return the array directly
        return aryOut

    else:
        # Put column with the indices of model-parameter-combinations into the
        # output array (in order to be able to put the pRF model time courses
        # into the correct order after the parallelised function):
        lstOut = [idxPrc,
                  aryOut]

        # Put output to queue:
        queOut.put(lstOut)