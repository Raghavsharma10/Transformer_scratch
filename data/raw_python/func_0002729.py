def funcFindPrfMltpPrdXVal(idxPrc,
                           aryFuncChnkTrn,
                           aryFuncChnkTst,
                           aryPrfMdlsTrnConv,
                           aryPrfMdlsTstConv,
                           aryMdls,
                           queOut):
    """
    Function for finding best pRF model for voxel time course.
    This function should be used if there are several predictors.
    """

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnkTrn.shape[0]

    # Number of volumes:
    varNumVolTrn = aryFuncChnkTrn.shape[2]
    varNumVolTst = aryFuncChnkTst.shape[2]

    # get number of cross validations
    varNumXval = aryPrfMdlsTrnConv.shape[2]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for temporary residuals values that are obtained during
    # the different loops of cross validation
    vecTmpResXVal = np.empty((varNumVoxChnk, varNumXval), dtype='float32')

    # Vector for best residual values.
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000.0)

    # Constant term for the model:
    vecConstTrn = np.ones((varNumVolTrn), dtype=np.float32)
    vecConstTst = np.ones((varNumVolTst), dtype=np.float32)

    # Change type to float 32:
    aryPrfMdlsTrnConv = aryPrfMdlsTrnConv.astype(np.float32)
    aryPrfMdlsTstConv = aryPrfMdlsTstConv.astype(np.float32)

    # Number of pRF models to fit:
    varNumMdls = len(aryMdls)

    # Prepare status indicator if this is the first of the parallel processes:
    if idxPrc == 0:

        # We create a status indicator for the time consuming pRF model finding
        # algorithm. Number of steps of the status indicator:
        varStsStpSze = 20

        # Vector with pRF values at which to give status feedback:
        vecStatPrf = np.linspace(0,
                                 varNumMdls,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrf = np.ceil(vecStatPrf)
        vecStatPrf = vecStatPrf.astype(int)

        # Vector with corresponding percentage values at which to give status
        # feedback:
        vecStatPrc = np.linspace(0,
                                 100,
                                 num=(varStsStpSze+1),
                                 endpoint=True)
        vecStatPrc = np.ceil(vecStatPrc)
        vecStatPrc = vecStatPrc.astype(int)

        # Counter for status indicator:
        varCntSts01 = 0
        varCntSts02 = 0

    # Loop through pRF models:
    for idxMdls in range(0, varNumMdls):

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Status indicator:
            if varCntSts02 == vecStatPrf[varCntSts01]:

                # Prepare status message:
                strStsMsg = ('---------Progress: ' +
                             str(vecStatPrc[varCntSts01]) +
                             ' % --- ' +
                             str(vecStatPrf[varCntSts01]) +
                             ' pRF models out of ' +
                             str(varNumMdls))

                print(strStsMsg)

                # Only increment counter if the last value has not been
                # reached yet:
                if varCntSts01 < varStsStpSze:
                    varCntSts01 = varCntSts01 + int(1)

        # Loop through different cross validations
        for idxXval in range(0, varNumXval):
            # Current pRF time course model:
            vecMdlTrn = aryPrfMdlsTrnConv[idxMdls, :, idxXval, :]
            vecMdlTst = aryPrfMdlsTstConv[idxMdls, :, idxXval, :]

            # We create a design matrix including the current pRF time
            # course model, and a constant term:
            aryDsgnTrn = np.vstack([vecMdlTrn,
                                    vecConstTrn]).T

            aryDsgnTst = np.vstack([vecMdlTst,
                                    vecConstTst]).T

            # Calculate the least-squares solution for all voxels
            # and get parameter estimates from the training fit
            aryTmpPrmEst = np.linalg.lstsq(aryDsgnTrn,
                                           aryFuncChnkTrn[:, idxXval, :].T)[0]
            # calculate predicted model fit based on training data
            aryTmpMdlTc = np.dot(aryDsgnTst, aryTmpPrmEst)
            # calculate residual sum of squares between test data and
            # predicted model fit based on training data
            vecTmpResXVal[:, idxXval] = np.sum(
                (np.subtract(aryFuncChnkTst[:, idxXval, :].T,
                             aryTmpMdlTc))**2, axis=0)

        vecTmpRes = np.mean(vecTmpResXVal, axis=1)
        # Check whether current residuals are lower than previously
        # calculated ones:
        vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

        # Replace best x and y position values, and SD values.
        vecBstXpos[vecLgcTmpRes] = aryMdls[idxMdls][0]
        vecBstYpos[vecLgcTmpRes] = aryMdls[idxMdls][1]
        vecBstSd[vecLgcTmpRes] = aryMdls[idxMdls][2]

        # Replace best residual values:
        vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              ]

    queOut.put(lstOut)