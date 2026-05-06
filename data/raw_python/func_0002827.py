def crt_prf_tc(aryNrlTc, varNumVol, varTr, varTmpOvsmpl, switchHrfSet,
               tplPngSize, varPar, dctPrm=None, lgcPrint=True):
    """Convolve every neural time course with HRF function.

    Parameters
    ----------
    aryNrlTc : 4d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_vol]
        Temporally upsampled neural time course models.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varTmpOvsmpl : int, positive
        Factor by which the data hs been temporally upsampled.
    switchHrfSet : int, (1, 2, 3)
        Switch to determine which hrf basis functions are used
    tplPngSize : tuple
        Pixel dimensions of the visual space (width, height).
    varPar : int, positive
        Number of cores for multi-processing.
    dctPrm : dictionary, default None
        Dictionary with customized hrf parameters. If this is None, default
        hrf parameters will be used.
    lgcPrint: boolean, default True
        Should print messages be sent to user?

    Returns
    -------
    aryNrlTcConv : 5d numpy array,
                   shape [n_x_pos, n_y_pos, n_sd, n_hrf_bases, varNumVol]
        Neural time courses convolved with HRF basis functions

    """

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # If necessary, adjust the input such that input is 2D, with last dim time
    tplInpShp = aryNrlTc.shape
    aryNrlTc = np.reshape(aryNrlTc, (-1, aryNrlTc.shape[-1]))

    if varPar == 1:
        # if the number of cores requested by the user is equal to 1,
        # we save the overhead of multiprocessing by calling aryMdlCndRsp
        # directly
        aryNrlTcConv = cnvl_tc(0, aryNrlTc, lstHrf, varTr,
                               varNumVol, varTmpOvsmpl, None, dctPrm=dctPrm)

    else:
        # Put input data into chunks:
        lstNrlTc = np.array_split(aryNrlTc, varPar)

        # Create a queue to put the results in:
        queOut = mp.Queue()

        # Empty list for processes:
        lstPrcs = [None] * varPar

        # Empty list for results of parallel processes:
        lstConv = [None] * varPar
        if lgcPrint:
            print('------------Running parallel processes')

        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvl_tc,
                                         args=(idxPrc,
                                               lstNrlTc[idxPrc],
                                               lstHrf,
                                               varTr,
                                               varNumVol,
                                               varTmpOvsmpl,
                                               queOut),
                                         kwargs={'dctPrm': dctPrm},
                                         )

            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

        # Start processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].start()

        # Collect results from queue:
        for idxPrc in range(0, varPar):
            lstConv[idxPrc] = queOut.get(True)

        # Join processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc].join()
        if lgcPrint:
            print('------------Collecting results from parallel processes')
        # Put output into correct order:
        lstConv = sorted(lstConv)
        # Concatenate convolved pixel time courses (into the same order
        aryNrlTcConv = np.zeros((0, switchHrfSet, varNumVol), dtype=np.float32)
        for idxRes in range(0, varPar):
            aryNrlTcConv = np.concatenate((aryNrlTcConv, lstConv[idxRes][1]),
                                          axis=0)
        # clean up
        del(aryNrlTc)
        del(lstConv)

    # Reshape results:
    tplOutShp = tplInpShp[:-1] + (len(lstHrf), ) + (varNumVol, )

    # Return:
    return np.reshape(aryNrlTcConv, tplOutShp).astype(np.float32)