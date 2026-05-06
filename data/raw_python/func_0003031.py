def cnvl_tc(idxPrc, aryPrfTcChunk, lstHrf, varTr, varNumVol, varTmpOvsmpl,
            queOut, varHrfLen=32., dctPrm=None):
    """Convolution of time courses with HRF model.

    Parameters
    ----------
    idxPrc : int, positive
        Process ID of the process calling this function (for CPU
        multi-threading). In GPU version, this parameter is 0 (just one thread
        on CPU).
    aryPrfTcChunk : np.array
        2D array with model time course to be convolved with HRF.
    lstHrf : list
        List containing the different HRF functions.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.
    varHrfLen : float, positive, default=32
        Length of the HRF time course in seconds.
    dctPrm : dictionary, default None
        Dictionary with customized hrf parameters. If this is None, default
        hrf parameters will be used.

    Returns
    -------
    lstOut : list
        int, positive : Process ID of the process calling this function.
        2D np.array, float16 : Model time course convolved with HRF.

    References:
    -----
    [1] https://github.com/fabianp/hrf_estimation

    """

    # Adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryPrfTcChunk.shape
    aryPrfTcChunk = aryPrfTcChunk.reshape((-1, aryPrfTcChunk.shape[-1]))

    # Prepare list to collect hrf basis functions
    lstBse = []
    # Prepare array that contains time intervals
    aryTme = np.linspace(0, varHrfLen, (varHrfLen // varTr) * varTmpOvsmpl)
    for fnHrf in lstHrf:
        # If hrf parameter dictionary is None, run with default parameters
        if dctPrm is None:
            vecTmpBse = fnHrf(aryTme)
        # Otherwise, run with custom parameters
        else:
            vecTmpBse = fnHrf(aryTme, **dctPrm)
        # Normalise HRF so that the sum of values is 1 (see FSL)
        # otherwise, after convolution values for predictors are very high
        vecTmpBse = np.divide(vecTmpBse, np.sum(vecTmpBse))

        lstBse.append(vecTmpBse)

    # Get frame times, i.e. start point of every volume in seconds
    vecFrms = np.arange(0, varTr * varNumVol, varTr)
    # Get supersampled frames times, i.e. start point of every volume in
    # upsampled res, since convolution takes place in temp. upsampled space
    vecFrmTms = np.arange(0, varTr * varNumVol, varTr / varTmpOvsmpl)

    # Prepare an empty array for ouput
    aryConv = np.zeros((aryPrfTcChunk.shape[0], len(lstHrf), varNumVol),
                       dtype=np.float16)
    # Each time course is convolved with the HRF separately, because the
    # numpy convolution function can only be used on one-dimensional data.
    # Thus, we have to loop through time courses:
    for idxTc in range(0, aryConv.shape[0]):

        # Extract the current time course (already in upsampled space):
        vecTcUps = aryPrfTcChunk[idxTc, :]

        # *** convolve
        for indBase, base in enumerate(lstBse):
            # Make sure base and vecTcUps are float64 to avoid overflow
            base = base.astype(np.float64)
            vecTcUps = vecTcUps.astype(np.float64)
            # Perform the convolution (previously: np.convolve)
            col = fftconvolve(base, vecTcUps, mode='full')[:vecTcUps.size]
            # Get function for downsampling
            f = interp1d(vecFrmTms, col)
            # Downsample to original resoltuion to match res of data
            # take the value from the centre of each volume's period (see FSL)
            aryConv[idxTc, indBase, :] = f(vecFrms + varTr/2.
                                           ).astype(np.float16)

    # Determine output shape
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (varNumVol, )

    if queOut is None:
        # if user is not using multiprocessing, return the array directly
        return aryConv.reshape(tplOutShp)

    else:
        # Create list containing the convolved timecourses, and the process ID:
        lstOut = [idxPrc,
                  aryConv.reshape(tplOutShp)]

        # Put output to queue:
        queOut.put(lstOut)