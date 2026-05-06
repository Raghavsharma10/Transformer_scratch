def cnvlTc(idxPrc,
           aryPrfTcChunk,
           lstHrf,
           varTr,
           varNumVol,
           queOut,
           varOvsmpl=10,
           varHrfLen=32,
           ):
    """
    Convolution of time courses with HRF model.
    """

    # *** prepare hrf time courses for convolution
    print("---------Process " + str(idxPrc) +
          ": Prepare hrf time courses for convolution")
    # get frame times, i.e. start point of every volume in seconds
    vecFrms = np.arange(0, varTr * varNumVol, varTr)
    # get supersampled frames times, i.e. start point of every volume in
    # seconds, since convolution takes place in temp. upsampled space
    vecFrmTms = np.arange(0, varTr * varNumVol, varTr / varOvsmpl)
    # get resolution of supersampled frame times
    varRes = varTr / float(varOvsmpl)

    # prepare empty list that will contain the arrays with hrf time courses
    lstBse = []
    for hrfFn in lstHrf:
        # needs to be a multiple of oversample
        vecTmpBse = hrfFn(np.linspace(0, varHrfLen,
                                      (varHrfLen // varTr) * varOvsmpl))
        lstBse.append(vecTmpBse)

    # *** prepare pixel time courses for convolution
    print("---------Process " + str(idxPrc) +
          ": Prepare pixel time courses for convolution")

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryPrfTcChunk.shape
    aryPrfTcChunk = aryPrfTcChunk.reshape((-1, aryPrfTcChunk.shape[-1]))

    # Prepare an empty array for ouput
    aryConv = np.zeros((aryPrfTcChunk.shape[0], len(lstHrf),
                        aryPrfTcChunk.shape[1]))
    print("---------Process " + str(idxPrc) +
          ": Convolve")
    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course:
        vecTc = aryPrfTcChunk[idxTc, :]

        # upsample the pixel time course, so that it matches the hrf time crs
        vecTcUps = np.zeros(int(varNumVol * varTr/varRes))
        vecOns = vecFrms[vecTc.astype(bool)]
        vecInd = np.round(vecOns / varRes).astype(np.int)
        vecTcUps[vecInd] = 1.

        # *** convolve
        for indBase, base in enumerate(lstBse):
            # perform the convolution
            col = np.convolve(base, vecTcUps, mode='full')[:vecTcUps.size]
            # get function for downsampling
            f = interp1d(vecFrmTms, col)
            # downsample to original space and assign to ary
            aryConv[idxTc, indBase, :] = f(vecFrms)

    # determine output shape
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (tplInpShp[-1], )

    # Create list containing the convolved timecourses, and the process ID:
    lstOut = [idxPrc,
              aryConv.reshape(tplOutShp)]

    # Put output to queue:
    queOut.put(lstOut)