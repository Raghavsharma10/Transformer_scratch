def crt_mdl_prms(tplPngSize, varNum1, varExtXmin,  varExtXmax, varNum2,
                 varExtYmin, varExtYmax, varNumPrfSizes, varPrfStdMin,
                 varPrfStdMax, kwUnt='pix', kwCrd='crt'):
    """Create an array with all possible model parameter combinations

    Parameters
    ----------
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space (width, height).
    varNum1 : int, positive
        Number of x-positions to model
    varExtXmin : float
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float
        Extent of visual space from centre in positive x-direction (width)
    varNum2 : float, positive
        Number of y-positions to model.
    varExtYmin : int
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float
        Extent of visual space from centre in positive y-direction (height)
    varNumPrfSizes : int, positive
        Number of pRF sizes to model.
    varPrfStdMin : float, positive
        Minimum pRF model size (standard deviation of 2D Gaussian)
    varPrfStdMax : float, positive
        Maximum pRF model size (standard deviation of 2D Gaussian)
    kwUnt: str
        Keyword to set the unit for model parameter combinations; model
        parameters can be in pixels ["pix"] or degrees of visual angles ["deg"]
    kwCrd: str
        Keyword to set the coordinate system for model parameter combinations;
        parameters can be in cartesian ["crt"] or polar ["pol"] coordinates

    Returns
    -------
    aryMdlParams : 2d numpy array, shape [n_x_pos*n_y_pos*n_sd, 3]
        Model parameters (x, y, sigma) for all models.

    """

    # Number of pRF models to be created (i.e. number of possible
    # combinations of x-position, y-position, and standard deviation):
    varNumMdls = varNum1 * varNum2 * varNumPrfSizes

    # Array for the x-position, y-position, and standard deviations for
    # which pRF model time courses are going to be created, where the
    # columns correspond to: (1) the x-position, (2) the y-position, and
    # (3) the standard deviation. The parameters are in units of the
    # upsampled visual space.
    aryMdlParams = np.zeros((varNumMdls, 3), dtype=np.float32)

    # Counter for parameter array:
    varCntMdlPrms = 0

    if kwCrd == 'crt':

        # Vector with the moddeled x-positions of the pRFs:
        vecX = np.linspace(varExtXmin, varExtXmax, varNum1, endpoint=True)

        # Vector with the moddeled y-positions of the pRFs:
        vecY = np.linspace(varExtYmin, varExtYmax, varNum2, endpoint=True)

        # Vector with standard deviations pRF models (in degree of vis angle):
        vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                               endpoint=True)

        if kwUnt == 'deg':
            # since parameters are already in degrees of visual angle,
            # we do nothing
            pass

        elif kwUnt == 'pix':
            # convert parameters to pixels
            vecX, vecY, vecPrfSd = rmp_deg_pixel_xys(vecX, vecY, vecPrfSd,
                                                     tplPngSize, varExtXmin,
                                                     varExtXmax, varExtYmin,
                                                     varExtYmax)

        else:
            print('Unknown keyword provided for possible model parameter ' +
                  'combinations: should be either pix or deg')

        # Put all combinations of x-position, y-position, and standard
        # deviations into the array:

        # Loop through x-positions:
        for idxX in range(0, varNum1):

            # Loop through y-positions:
            for idxY in range(0, varNum2):

                # Loop through standard deviations (of Gaussian pRF models):
                for idxSd in range(0, varNumPrfSizes):

                    # Place index and parameters in array:
                    aryMdlParams[varCntMdlPrms, 0] = vecX[idxX]
                    aryMdlParams[varCntMdlPrms, 1] = vecY[idxY]
                    aryMdlParams[varCntMdlPrms, 2] = vecPrfSd[idxSd]

                    # Increment parameter index:
                    varCntMdlPrms += 1

    elif kwCrd == 'pol':

        # Vector with the radial position:
        vecRad = np.linspace(0.0, varExtXmax, varNum1, endpoint=True)

        # Vector with the angular position:
        vecTht = np.linspace(0.0, 2*np.pi, varNum2, endpoint=False)

        # Get all possible combinations on the grid, using matrix indexing ij
        # of output
        aryRad, aryTht = np.meshgrid(vecRad, vecTht, indexing='ij')

        # Flatten arrays to be able to combine them with meshgrid
        vecRad = aryRad.flatten()
        vecTht = aryTht.flatten()

        # Convert from polar to cartesian
        vecX, vecY = map_pol_to_crt(vecTht, vecRad)

        # Vector with standard deviations pRF models (in degree of vis angle):
        vecPrfSd = np.linspace(varPrfStdMin, varPrfStdMax, varNumPrfSizes,
                               endpoint=True)

        if kwUnt == 'deg':
            # since parameters are already in degrees of visual angle,
            # we do nothing
            pass

        elif kwUnt == 'pix':
            # convert parameters to pixels
            vecX, vecY, vecPrfSd = rmp_deg_pixel_xys(vecX, vecY, vecPrfSd,
                                                     tplPngSize, varExtXmin,
                                                     varExtXmax, varExtYmin,
                                                     varExtYmax)
        # Put all combinations of x-position, y-position, and standard
        # deviations into the array:

        # Loop through x-positions:
        for idxXY in range(0, varNum1*varNum2):

            # Loop through standard deviations (of Gaussian pRF models):
            for idxSd in range(0, varNumPrfSizes):

                # Place index and parameters in array:
                aryMdlParams[varCntMdlPrms, 0] = vecX[idxXY]
                aryMdlParams[varCntMdlPrms, 1] = vecY[idxXY]
                aryMdlParams[varCntMdlPrms, 2] = vecPrfSd[idxSd]

                # Increment parameter index:
                varCntMdlPrms += 1

    else:
        print('Unknown keyword provided for coordinate system for model ' +
              'parameter combinations: should be either crt or pol')

    return aryMdlParams