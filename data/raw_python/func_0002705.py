def cnvlTcOld(idxPrc,
              aryPrfTcChunk,
              varTr,
              varNumVol,
              queOut):
    """
    Old version:
    Convolution of time courses with one canonical HRF model.
    """
    # Create 'canonical' HRF time course model:
    vecHrf = funcHrf(varNumVol, varTr)

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryPrfTcChunk.shape
    aryPrfTcChunk = aryPrfTcChunk.reshape((-1, aryPrfTcChunk.shape[-1]))

    # Prepare an empty array for ouput
    aryConv = np.zeros(np.shape(aryPrfTcChunk))

    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc, vecTc in enumerate(aryPrfTcChunk):

        # In order to avoid an artefact at the end of the time series, we have
        # to concatenate an empty array to both the design matrix and the HRF
        # model before convolution.
        vecTc = np.append(vecTc, np.zeros(100))
        vecHrf = np.append(vecHrf, np.zeros(100))

        # Convolve design matrix with HRF model:
        aryConv[idxTc, :] = np.convolve(vecTc, vecHrf,
                                        mode='full')[:varNumVol]

    # determine output shape
    tplOutShp = tplInpShp[:-1] + (1, ) + (tplInpShp[-1], )

    # Create list containing the convolved timecourses, and the process ID:
    lstOut = [idxPrc,
              aryConv.reshape(tplOutShp)]

    # Put output to queue:
    queOut.put(lstOut)