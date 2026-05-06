def rsmplInHighRes(aryBoxCarConv,
                   tplPngSize,
                   tplVslSpcHighSze,
                   varNumMtDrctn,
                   varNumVol):
    """Resample pixel-time courses in high-res visual space.

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
    # Array for super-sampled pixel-time courses:
    aryBoxCarConvHigh = np.zeros((tplVslSpcHighSze[0],
                                  tplVslSpcHighSze[1],
                                  varNumMtDrctn,
                                  varNumVol))

    # Loop through volumes:
    for idxMtn in range(0, varNumMtDrctn):

        for idxVol in range(0, varNumVol):

            # Range for the coordinates:
            vecRange = np.arange(0, tplPngSize[0])

            # The following array describes the coordinates of the pixels in
            # the flattened array (i.e. "vecOrigPixVal"). In other words, these
            # are the row and column coordinates of the original pizel values.
            crd2, crd1 = np.meshgrid(vecRange, vecRange)
            aryOrixPixCoo = np.column_stack((crd1.flatten(), crd2.flatten()))

            # The following vector will contain the actual original pixel
            # values:

            vecOrigPixVal = aryBoxCarConv[:, :, idxMtn, idxVol]
            vecOrigPixVal = vecOrigPixVal.flatten()

            # The sampling interval for the creation of the super-sampled pixel
            # data (complex numbers are used as a convention for inclusive
            # intervals in "np.mgrid()").:

            varStpSzeX = np.complex(tplVslSpcHighSze[0])
            varStpSzeY = np.complex(tplVslSpcHighSze[1])

            # The following grid has the coordinates of the points at which we
            # would like to re-sample the pixel data:
            aryPixGridX, aryPixGridY = np.mgrid[0:tplPngSize[0]:varStpSzeX,
                                                0:tplPngSize[1]:varStpSzeY]

            # The actual resampling:
            aryResampled = griddata(aryOrixPixCoo,
                                    vecOrigPixVal,
                                    (aryPixGridX, aryPixGridY),
                                    method='nearest')

            # Put super-sampled pixel time courses into array:
            aryBoxCarConvHigh[:, :, idxMtn, idxVol] = aryResampled

    return aryBoxCarConvHigh