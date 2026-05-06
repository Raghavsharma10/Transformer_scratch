def cnvlPwBoxCarFn(aryNrlTc, varNumVol, varTr, tplPngSize, varNumMtDrctn,
                   switchHrfSet, lgcOldSchoolHrf, varPar,):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    aryNrlTc : 5d numpy array, shape [n_x_pos, n_y_pos, n_sd, n_mtn_dir, n_vol]
        Description of input 1.
    varNumVol : float, positive
        Description of input 2.
    varTr : float, positive
        Description of input 1.
    tplPngSize : tuple
        Description of input 1.
    varNumMtDrctn : int, positive
        Description of input 1.
    switchHrfSet :
        Description of input 1.
    lgcOldSchoolHrf : int, positive
        Description of input 1.
    varPar : int, positive
        Description of input 1.
    Returns
    -------
    data : 2d numpy array, shape [n_samples, n_measurements]
        Closed data.
    Reference
    ---------
    [1]
    """
    print('------Convolve every pixel box car function with hrf function(s)')

    # Create hrf time course function:
    if switchHrfSet == 3:
        lstHrf = [spmt, dspmt, ddspmt]
    elif switchHrfSet == 2:
        lstHrf = [spmt, dspmt]
    elif switchHrfSet == 1:
        lstHrf = [spmt]

    # adjust the input, if necessary, such that input is 2D, with last dim time
    tplInpShp = aryNrlTc.shape
    aryNrlTc = np.reshape(aryNrlTc, (-1, aryNrlTc.shape[-1]))

    # Put input data into chunks:
    lstNrlTc = np.array_split(aryNrlTc, varPar)

    # Create a queue to put the results in:
    queOut = mp.Queue()

    # Empty list for processes:
    lstPrcs = [None] * varPar

    # Empty list for results of parallel processes:
    lstConv = [None] * varPar

    print('---------Creating parallel processes')

    if lgcOldSchoolHrf:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTcOld,
                                         args=(idxPrc,
                                               lstNrlTc[idxPrc],
                                               varTr,
                                               varNumVol,
                                               queOut)
                                         )
            # Daemon (kills processes when exiting):
            lstPrcs[idxPrc].Daemon = True

    else:
        # Create processes:
        for idxPrc in range(0, varPar):
            lstPrcs[idxPrc] = mp.Process(target=cnvlTc,
                                         args=(idxPrc,
                                               lstNrlTc[idxPrc],
                                               lstHrf,
                                               varTr,
                                               varNumVol,
                                               queOut)
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

    print('---------Collecting results from parallel processes')
    # Put output into correct order:
    lstConv = sorted(lstConv)
    # Concatenate convolved pixel time courses (into the same order
    aryNrlTcConv = np.zeros((0, switchHrfSet, varNumVol))
    for idxRes in range(0, varPar):
        aryNrlTcConv = np.concatenate((aryNrlTcConv, lstConv[idxRes][1]),
                                      axis=0)
    # clean up
    del(aryNrlTc)
    del(lstConv)

    # Reshape results:
    tplOutShp = tplInpShp[:-2] + (varNumMtDrctn * len(lstHrf), ) + \
        (tplInpShp[-1], )

    # Return:
    return np.reshape(aryNrlTcConv, tplOutShp)