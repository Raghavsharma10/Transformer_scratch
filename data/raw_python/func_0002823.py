def rmp_pixel_deg_xys(vecX, vecY, vecPrfSd, tplPngSize,
                      varExtXmin, varExtXmax, varExtYmin, varExtYmax):
    """Remap x, y, sigma parameters from pixel to degree.

    Parameters
    ----------
    vecX : 1D numpy array
        Array with possible x parametrs in pixels
    vecY : 1D numpy array
        Array with possible y parametrs in pixels
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in pixels
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space in pixel (width, height).
    varExtXmin : float
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float
        Extent of visual space from centre in positive x-direction (width)
    varExtYmin : int
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float
        Extent of visual space from centre in positive y-direction (height)

    Returns
    -------
    vecX : 1D numpy array
        Array with possible x parametrs in degree
    vecY : 1D numpy array
        Array with possible y parametrs in degree
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in degree

    """

    # Remap modelled x-positions of the pRFs:
    vecXdgr = rmp_rng(vecX, varExtXmin, varExtXmax, varOldThrMin=0.0,
                      varOldAbsMax=(tplPngSize[0] - 1))

    # Remap modelled y-positions of the pRFs:
    vecYdgr = rmp_rng(vecY, varExtYmin, varExtYmax, varOldThrMin=0.0,
                      varOldAbsMax=(tplPngSize[1] - 1))

    # We calculate the scaling factor from pixels to degrees of visual angle to
    # separately for the x- and the y-directions (the two should be the same).
    varPix2DgrX = np.divide((varExtXmax - varExtXmin), tplPngSize[0])
    varPix2DgrY = np.divide((varExtYmax - varExtYmin), tplPngSize[1])

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in ' + \
        'stimulus space (in pixels) do not agree'
    assert 0.5 > np.absolute((varPix2DgrX - varPix2DgrY)), strErrMsg

    # Convert prf sizes from degrees of visual angles to pixel
    vecPrfSdDgr = np.multiply(vecPrfSd, varPix2DgrX)

    # Return new values.
    return vecXdgr, vecYdgr, vecPrfSdDgr