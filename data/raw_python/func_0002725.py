def funcConvPar(aryDm,
                vecHrf,
                varNumVol):

    """
    Function for convolution of pixel-wise 'design matrix' with HRF model.
    """
    # In order to avoid an artefact at the end of the time series, we have to
    # concatenate an empty array to both the design matrix and the HRF model
    # before convolution.
    aryDm = np.concatenate((aryDm, np.zeros((aryDm.shape[0], 100))), axis=1)
    vecHrf = np.concatenate((vecHrf, np.zeros((100,))))

    aryDmConv = np.empty((aryDm.shape[0], varNumVol))
    for idx in range(0, aryDm.shape[0]):
        vecDm = aryDm[idx, :]
        # Convolve design matrix with HRF model:
        aryDmConv[idx, :] = np.convolve(vecDm, vecHrf,
                                        mode='full')[:varNumVol]
    return aryDmConv