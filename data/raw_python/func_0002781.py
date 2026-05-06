def crtPrfNrlTc(aryBoxCar, varNumMtDrctn, varNumVol, tplPngSize, varNumX,
                varExtXmin,  varExtXmax, varNumY, varExtYmin, varExtYmax,
                varNumPrfSizes, varPrfStdMin, varPrfStdMax, varPar):
    """Create neural model time courses from pixel-wise boxcar functions.

    Parameters
    ----------
    aryBoxCar : 4d numpy array, shape [n_x_pix, n_y_pix, n_mtn_dir, n_vol]
        Description of input 1.
    varNumMtDrctn : float, positive
        Description of input 2.
    varNumVol : float, positive
        Description of input 2.
    tplPngSize : tuple
        Description of input 2.
    varNumX : float, positive
        Description of input 2.
    varExtXmin : float, positive
        Description of input 2.
    varExtXmax : float, positive
        Description of input 2.
    varNumY : float, positive
        Description of input 2.
    varExtYmin : float, positive
        Description of input 2.
    varExtYmax : float, positive
        Description of input 2.
    varNumPrfSizes : float, positive
        Description of input 2.
    varPrfStdMin : float, positive
        Description of input 2.
    varPrfStdMax : float, positive
        Description of input 2.
    varPar : float, positive
        Description of input 2.
    Returns
    -------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Create neural time course models')

    # Vector with the x-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecX = np.linspace(0, (tplPngSize[0] - 1), varNumX, endpoint=True)

    # Vector with the y-indicies of the positions in the super-sampled visual
    # space at which to create pRF models.
    vecY = np.linspace(0, (tplPngSize[1] - 1), varNumY, endpoint=True)

    # We calculate the scaling factor from degrees of visual angle to pixels
    # separately for the x- and the y-directions (the two should be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in stimulus ' + \
        'space (in degrees of visual angle) and the ratio of X and Y ' + \
        'dimensions in the upsampled visual space do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Vector with pRF sizes to be modelled (still in degree of visual angle):
    vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                           endpoint=True)

    # We multiply the vector containing pRF sizes with the scaling factors.
    # Now the vector with the pRF sizes can be used directly for creation of
    # Gaussian pRF models in visual space.
    vecPrfSd = np.multiply(vecPrfSd, varDgr2PixX)

    # Number of pRF models to be created (i.e. number of possible combinations
    # of x-position, y-position, and standard deviation):
    varNumMdls = varNumX * varNumY * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for which
    # pRF model time courses are going to be created, where the columns
    # correspond to: (0) an index starting from zero, (1) the x-position, (2)
    # the y-position, and (3) the standard deviation. The parameters are in
    # units of the upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 4))

    # Counter for parameter array:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = varCntMdlPrms
                aryMdlParams[varCntMdlPrms, 1] = vecX[idxX]
                aryMdlParams[varCntMdlPrms, 2] = vecY[idxY]
                aryMdlParams[varCntMdlPrms, 3] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    # The long array with all the combinations of model parameters is put into
    # separate chunks for parallelisation, using a list of arrays.
    lstMdlParams = np.array_split(aryMdlParams, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for results from parallel processes (for pRF model time course
    # results):
    lstPrfTc = [None] * varPar

    # Empty list for processes:
    lstPrcs = [None] * varPar

    print('---------Creating parallel processes')

    # Create processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc] = mp.Process(target=cnvlGauss2D,
                                     args=(idxPrc, aryBoxCar,
                                           lstMdlParams[idxPrc], tplPngSize,
                                           varNumVol, queOut)
                                     )
        # Daemon (kills processes when exiting):
        lstPrcs[idxPrc].Daemon = True

    # Start processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].start()

    # Collect results from queue:
    for idxPrc in range(0, varPar):
        lstPrfTc[idxPrc] = queOut.get(True)

    # Join processes:
    for idxPrc in range(0, varPar):
        lstPrcs[idxPrc].join()

    print('---------Collecting results from parallel processes')
    # Put output arrays from parallel process into one big array
    lstPrfTc = sorted(lstPrfTc)
    aryPrfTc = np.empty((0, varNumMtDrctn, varNumVol))
    for idx in range(0, varPar):
        aryPrfTc = np.concatenate((aryPrfTc, lstPrfTc[idx][1]), axis=0)

    # check that all the models were collected correctly
    assert aryPrfTc.shape[0] == varNumMdls

    # Clean up:
    del(aryMdlParams)
    del(lstMdlParams)
    del(lstPrfTc)

    # Array representing the low-resolution visual space, of the form
    # aryPrfTc[x-position, y-position, pRF-size, varNum Vol], which will hold
    # the pRF model time courses.
    aryNrlTc = np.zeros([varNumX, varNumY, varNumPrfSizes, varNumMtDrctn,
                         varNumVol])

    # We use the same loop structure for organising the pRF model time courses
    # that we used for creating the parameter array. Counter:
    varCntMdlPrms = 0

    # Put all combinations of x-position, y-position, and standard deviations
    # into the array:

    # Loop through x-positions:
    for idxX in range(0, varNumX):

        # Loop through y-positions:
        for idxY in range(0, varNumY):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Put the pRF model time course into its correct position in
                # the 4D array, leaving out the first column (which contains
                # the index):
                aryNrlTc[idxX, idxY, idxSd, :, :] = aryPrfTc[
                    varCntMdlPrms, :, :]

                # Increment parameter index:
                varCntMdlPrms = varCntMdlPrms + 1

    return aryNrlTc