def funcNrlTcMotPred(idxPrc,
                     varPixX,
                     varPixY,
                     NrlMdlChunk,
                     varNumTP,
                     aryBoxCar,  # aryCond
                     path,
                     varNumNrlMdls,
                     varNumMtDrctn,
                     varPar,
                     queOut):
    """
    Function for creating neural time course models.
    This function should be used to create neural models if different
    predictors for every motion direction are included.
    """

#    # if hd5 method is used: open file for reading
#    filename = 'aryBoxCar' + str(idxPrc) + '.hdf5'
#    hdf5_path = os.path.join(path, filename)
#    fileH = tables.openFile(hdf5_path, mode='r')

    # Output array with pRF model time courses at all modelled standard
    # deviations for current pixel position:
    aryOut = np.empty((len(NrlMdlChunk), varNumTP, varNumMtDrctn),
                      dtype='float32')

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 1:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Number of pRF models to fit:
        varNumLoops = varNumNrlMdls/varPar

        # Vector with pRF values at which to give status feedback:
        vecStatus = np.linspace(0,
                                varNumLoops,
                                num=(varStsStpSze+1),
                                endpoint=True)
        vecStatus = np.ceil(vecStatus)
        vecStatus = vecStatus.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatusPrc = np.linspace(0,
                                   100,
                                   num=(varStsStpSze+1),
                                   endpoint=True)
        vecStatusPrc = np.ceil(vecStatusPrc)
        vecStatusPrc = vecStatusPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through all Gauss parameters that are in this chunk
    for idx, NrlMdlTrpl in enumerate(NrlMdlChunk):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:

            # Status indicator:
            if varCntSts02 == vecStatus[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatusPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatus[varCntSts01]) +
                             ' loops out of ' +
                             str(varNumLoops))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # x pos of Gauss model: NrlMdlTrpl[0]
        # y pos of Gauss model: NrlMdlTrpl[1]
        # std of Gauss model: NrlMdlTrpl[2]
        # index of tng crv model: NrlMdlTrpl[3]
        varTmpX = int(np.around(NrlMdlTrpl[0], 0))
        varTmpY = int(np.around(NrlMdlTrpl[1], 0))

        # Create pRF model (2D):
        aryGauss = funcGauss2D(varPixX,
                               varPixY,
                               varTmpX,
                               varTmpY,
                               NrlMdlTrpl[2])

        # Multiply pixel-wise box car model with Gaussian pRF models:
        aryNrlTcTmp = np.multiply(aryBoxCar, aryGauss[:, :, None, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the
        # neural time course model (i.e. not yet scaled for the size of
        # the pRF).
        aryNrlTcTmp = np.sum(aryNrlTcTmp, axis=(0, 1))

        # Normalise the nrl time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point,
        # or, in other words, the neural time course model.
        aryNrlTcTmp = np.divide(aryNrlTcTmp,
                                np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idx, :, :] = aryNrlTcTmp

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 1:
            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              aryOut,
              ]

    queOut.put(lstOut)