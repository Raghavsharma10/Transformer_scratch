def funcFindPrf(idxPrc,
                aryFuncChnk,
                aryPrfTc,
                aryMdls,
                queOut):

    """
    Function for finding best pRF model for voxel time course.
    This function should be used if there is only one predictor.
    """

    # Number of voxels to be fitted in this chunk:
    varNumVoxChnk = aryFuncChnk.shape[0]

    # Number of volumes:
    varNumVol = aryFuncChnk.shape[1]

    # Vectors for pRF finding results [number-of-voxels times one]:
    vecBstXpos = np.zeros(varNumVoxChnk)
    vecBstYpos = np.zeros(varNumVoxChnk)
    vecBstSd = np.zeros(varNumVoxChnk)
    # vecBstR2 = np.zeros(varNumVoxChnk)

    # Vector for best R-square value. For each model fit, the R-square value is
    # compared to this, and updated if it is lower than the best-fitting
    # solution so far. We initialise with an arbitrary, high value
    vecBstRes = np.add(np.zeros(varNumVoxChnk),
                       100000.0)

    # We reshape the voxel time courses, so that time goes down the column,
    # i.e. from top to bottom.
    aryFuncChnk = aryFuncChnk.T

    # Constant term for the model:
    vecConst = np.ones((varNumVol), dtype=np.float32)

    # Change type to float 32:
    aryFuncChnk = aryFuncChnk.astype(np.float32)
    aryPrfTc = aryPrfTc.astype(np.float32)

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

        # Current pRF time course model:
        vecMdlTc = aryPrfTc[idxMdls, :].flatten()

        # We create a design matrix including the current pRF time
        # course model, and a constant term:
        aryDsgn = np.vstack([vecMdlTc,
                             vecConst]).T

        # Calculation of the ratio of the explained variance (R square)
        # for the current model for all voxel time courses.

#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- START')
#                varTmeTmp01 = time.time()

        # Change type to float32:
        # aryDsgn = aryDsgn.astype(np.float32)

        # Calculate the least-squares solution for all voxels:
        vecTmpRes = np.linalg.lstsq(aryDsgn, aryFuncChnk)[1]

#                varTmeTmp02 = time.time()
#                varTmeTmp03 = np.around((varTmeTmp02 - varTmeTmp01),
#                                        decimals=2)
#                print('------------np.linalg.lstsq on pRF: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- DONE elapsed time: ' +
#                      str(varTmeTmp03) +
#                      's')

        # Check whether current residuals are lower than previously
        # calculated ones:
        vecLgcTmpRes = np.less(vecTmpRes, vecBstRes)

        # Replace best x and y position values, and SD values.
        vecBstXpos[vecLgcTmpRes] = aryMdls[idxMdls][0]
        vecBstYpos[vecLgcTmpRes] = aryMdls[idxMdls][1]
        vecBstSd[vecLgcTmpRes] = aryMdls[idxMdls][2]

        # Replace best residual values:
        vecBstRes[vecLgcTmpRes] = vecTmpRes[vecLgcTmpRes]

#                varTmeTmp04 = time.time()
#                varTmeTmp05 = np.around((varTmeTmp04 - varTmeTmp02),
#                                        decimals=2)
#                print('------------selection of best-fitting pRF model: ' +
#                      str(idxX) +
#                      'x ' +
#                      str(idxY) +
#                      'y ' +
#                      str(idxSd) +
#                      'z --- elapsed time: ' +
#                      str(varTmeTmp05) +
#                      's')

        # Status indicator (only used in the first of the parallel
        # processes):
        if idxPrc == 0:

            # Increment status indicator counter:
            varCntSts02 = varCntSts02 + 1

    # After finding the best fitting model for each voxel, we still have to
    # calculate the coefficient of determination (R-squared) for each voxel. We
    # start by calculating the total sum of squares (i.e. the deviation of the
    # data from the mean). The mean of each time course:
    vecFuncMean = np.mean(aryFuncChnk, axis=0)
    # Deviation from the mean for each datapoint:
    vecFuncDev = np.subtract(aryFuncChnk, vecFuncMean[None, :])
    # Sum of squares:
    vecSsTot = np.sum(np.power(vecFuncDev,
                               2.0),
                      axis=0)
    # Coefficient of determination:
    vecBstR2 = np.subtract(1.0,
                           np.divide(vecBstRes,
                                     vecSsTot))

    # Output list:
    lstOut = [idxPrc,
              vecBstXpos,
              vecBstYpos,
              vecBstSd,
              vecBstR2]

    queOut.put(lstOut)