def cmp_res_R2(lstRat, lstNiiNames, strPathOut, strPathMdl, lgcSveMdlTc=True,
               lgcDel=False, strNmeExt=''):
    """"Compare results for different exponents and create winner nii.

    Parameters
    ----------
    lstRat : list
        List of floats containing the ratios that were tested for surround
        suppression.
    lstNiiNames : list
        List of names of the different pRF maps (e.g. xpos, ypos, SD)
    strPathOut : string
        Path to the parent directory where the results should be saved.
    strPathMdl : string
        Path to the parent directory where pRF models should be saved.
    lgcDel : boolean
        Should model time courses be saved as npy file?
    lgcDel : boolean
        Should inbetween results (in form of nii files) be deleted?
    strNmeExt : string
        Extra name appendix to denominate experiment name. If undesidered,
        provide empty string.

    Notes
    -----
    [1] This function does not return any arrays but instead saves to disk.

    """

    print('---Compare results for different ratios')

    # Extract the index position for R2 and Betas map in lstNiiNames
    indPosR2 = [ind for ind, item in enumerate(lstNiiNames) if 'R2' in item]
    indPosBetas = [ind for ind, item in enumerate(lstNiiNames) if 'Betas' in
                   item]
    # Check that only one index was found
    msgError = 'More than one nii file was provided that could serve as R2 map'
    assert len(indPosR2) == 1, msgError
    assert len(indPosBetas) == 1, msgError
    # turn list int index
    indPosR2 = indPosR2[0]
    indPosBetas = indPosBetas[0]

    # Get the names of the nii files with in-between results
    lstCmpRes = []
    for indRat in range(len(lstRat)):
        # Get strExpSve
        strExpSve = '_' + str(lstRat[indRat])
        # If ratio is marked with 1.0, set empty string to find results.
        # 1.0 is the key for fitting without a surround.
        if lstRat[indRat] == 1.0:
            strExpSve = ''
        # Create full path names from nii file names and output path
        lstPthNames = [strPathOut + strNii + strNmeExt + strExpSve + '.nii.gz'
                       for strNii in lstNiiNames]
        # Append list to list that contains nii names for all exponents
        lstCmpRes.append(lstPthNames)

    print('------Find ratio that yielded highest R2 per voxel')

    # Initialize winner R2 map with R2 values from fit without surround
    aryWnrR2 = load_nii(lstCmpRes[0][indPosR2])[0]
    # Initialize ratio map with 1 where no-surround model was fit, otherwise 0
    aryRatMap = np.zeros(aryWnrR2.shape)
    aryRatMap[np.nonzero(aryWnrR2)] = 1.0

    # Loop over R2 maps to establish which exponents wins
    # Skip the first ratio, since this is the reference ratio (no surround)
    # and is reflected already in the initialized arrays - aryWnrR2 & aryRatMap
    for indRat, lstMaps in zip(lstRat[1:], lstCmpRes[1:]):
        # Load R2 map for this particular exponent
        aryTmpR2 = load_nii(lstMaps[indPosR2])[0]
        # Load beta values for this particular exponent
        aryTmpBetas = load_nii(lstMaps[indPosBetas])[0]
        # Get logical that tells us where current R2 map is greater than
        # previous ones
        aryLgcWnr = np.greater(aryTmpR2, aryWnrR2)
        # Get logical that tells us where the beta parameter estimate for the
        # centre is positive and the estimate for the surround is negative
        aryLgcCtrSur1 = np.logical_and(np.greater(aryTmpBetas[..., 0], 0.0),
                                       np.less(aryTmpBetas[..., 1], 0.0))
        # Get logical that tells us where the absolute beta parameter estimate
        # for the surround is less than beta parameter estimate for the center
        aryLgcCtrSur2 = np.less(np.abs(aryTmpBetas[..., 1]),
                                np.abs(aryTmpBetas[..., 0]))
        # Combine the two logicals
        aryLgcCtrSur = np.logical_and(aryLgcCtrSur1, aryLgcCtrSur2)
        # Combine logical for winner R2 and center-surround conditions
        aryLgcWnr = np.logical_and(aryLgcWnr, aryLgcCtrSur)
        # Replace values of R2, where current R2 map was greater
        aryWnrR2[aryLgcWnr] = np.copy(aryTmpR2[aryLgcWnr])
        # Remember the index of the exponent that gave rise to this new R2
        aryRatMap[aryLgcWnr] = indRat

    # Initialize list with winner maps. The winner maps are initialized with
    # the same shape as the maps that the last tested ratio maps had.
    lstRatMap = []
    for strPthMaps in lstCmpRes[-1]:
        lstRatMap.append(np.zeros(nb.load(strPthMaps).shape))

    # Compose other maps by assigning map value from the map that resulted from
    # the exponent that won for particular voxel
    for indRat, lstMaps in zip(lstRat, lstCmpRes):
        # Find out where this exponent won in terms of R2
        lgcWinnerMap = [aryRatMap == indRat][0]
        # Loop over all the maps
        for indMap, _ in enumerate(lstMaps):
            # Load map for this particular ratio
            aryTmpMap = load_nii(lstMaps[indMap])[0]
            # Handle exception: beta map will be 1D, if from ratio 1.0
            # In this case we want to make it 2D. In particular, the second
            # set of beta weights should be all zeros, so that later when
            # forming the model time course, the 2nd predictors contributes 0
            if indRat == 1.0 and indMap == indPosBetas:
                aryTmpMap = np.concatenate((aryTmpMap,
                                            np.zeros(aryTmpMap.shape)),
                                           axis=-1)
            # Load current winner map from array
            aryCrrWnrMap = np.copy(lstRatMap[indMap])
            # Assign values in temporary map to current winner map for voxels
            # where this ratio won
            aryCrrWnrMap[lgcWinnerMap] = np.copy(aryTmpMap[lgcWinnerMap])
            lstRatMap[indMap] = aryCrrWnrMap

    print('------Export results as nii')

    # Save winner maps as nii files
    # Get header and affine array
    hdrMsk, aryAff = load_nii(lstMaps[indPosR2])[1:]
    # Loop over all the maps
    for indMap, aryMap in enumerate(lstRatMap):
        # Create nii object for results:
        niiOut = nb.Nifti1Image(aryMap,
                                aryAff,
                                header=hdrMsk
                                )
        # Save nii:
        strTmp = strPathOut + '_supsur' + lstNiiNames[indMap] + strNmeExt + \
            '.nii.gz'
        nb.save(niiOut, strTmp)

    # Save map with best ratios as nii
    niiOut = nb.Nifti1Image(aryRatMap,
                            aryAff,
                            header=hdrMsk
                            )
    # Save nii:
    strTmp = strPathOut + '_supsur' + '_Ratios' + strNmeExt + '.nii.gz'
    nb.save(niiOut, strTmp)

    if lgcSveMdlTc:
        print('------Save model time courses/parameters/responses for ' +
              'centre and surround, across all ratios')
    
        # Get the names of the npy files with inbetween model responses
        lstCmpMdlRsp = []
        for indRat in range(len(lstRat)):
            # Get strExpSve
            strExpSve = '_' + str(lstRat[indRat])
            # If ratio is marked with 0, set empty string to find results.
            # This is the code for fitting without a surround.
            if lstRat[indRat] == 1.0:
                strExpSve = ''
            # Create full path names from npy file names and output path
            lstPthNames = [strPathMdl + strNpy + strNmeExt + strExpSve + '.npy'
                           for strNpy in ['', '_params', '_mdlRsp']]
            # Append list to list that contains nii names for all exponents
            lstCmpMdlRsp.append(lstPthNames)
    
        # Load tc/parameters/responses for different ratios, for now skip "0.0"
        # ratio because its tc/parameters/responses differs in shape
        lstPrfTcSur = []
        lstMdlParamsSur = []
        lstMdlRspSur = []
        for indNpy, lstNpy in enumerate(lstCmpMdlRsp[1:]):
            lstPrfTcSur.append(np.load(lstNpy[0]))
            lstMdlParamsSur.append(np.load(lstNpy[1]))
            lstMdlRspSur.append(np.load(lstNpy[2]))
        # Turn into arrays
        aryPrfTcSur = np.stack(lstPrfTcSur, axis=2)
        aryMdlParamsSur = np.stack(lstMdlParamsSur, axis=2)
        aryMdlRspSur = np.stack(lstMdlRspSur, axis=2)
    
        # Now handle the "1.0" ratio
        # Load the tc/parameters/responses of the "1.0" ratio
        aryPrfTc = np.load(lstCmpMdlRsp[0][0])
        aryMdlParams = np.load(lstCmpMdlRsp[0][1])
        aryMdlRsp = np.load(lstCmpMdlRsp[0][2])
        # Make 2nd row of time courses all zeros so they get no weight in lstsq
        aryPrfTc = np.concatenate((aryPrfTc, np.zeros(aryPrfTc.shape)), axis=1)
        # Make 2nd row of parameters the same as first row
        aryMdlParams = np.stack((aryMdlParams, aryMdlParams), axis=1)
        # Make 2nd row of responses all zeros so they get no weight in lstsq
        aryMdlRsp = np.stack((aryMdlRsp, np.zeros(aryMdlRsp.shape)), axis=1)
        # Add the "1.0" ratio to tc/parameters/responses of other ratios
        aryPrfTcSur = np.concatenate((np.expand_dims(aryPrfTc, axis=2),
                                      aryPrfTcSur), axis=2)
        aryMdlParamsSur = np.concatenate((np.expand_dims(aryMdlParams, axis=2),
                                         aryMdlParamsSur), axis=2)
        aryMdlRspSur = np.concatenate((np.expand_dims(aryMdlRsp, axis=2),
                                      aryMdlRspSur), axis=2)
    
        # Save parameters/response for centre and surround, for all ratios
        np.save(strPathMdl + '_supsur' + '', aryPrfTcSur)
        np.save(strPathMdl + '_supsur' + '_params', aryMdlParamsSur)
        np.save(strPathMdl + '_supsur' + '_mdlRsp', aryMdlRspSur)

    # Delete all the inbetween results, if desired by user, skip "0.0" ratio
    if lgcDel:
        lstCmpRes = [item for sublist in lstCmpRes[1:] for item in sublist]
        print('------Delete in-between results')
        for strMap in lstCmpRes[:]:
            os.remove(strMap)
        if lgcSveMdlTc:
            lstCmpMdlRsp = [item for sublist in lstCmpMdlRsp[1:] for item in
                            sublist]
            for strMap in lstCmpMdlRsp[:]:
                os.remove(strMap)