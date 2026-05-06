def loadPrsOrd(vecRunLngth, strPathPresOrd, vecVslStim):
    """Load presentation order of motion directions.

    Parameters
    ----------
    vecRunLngth : list
        Number of volumes in every run
    strPathPresOrd : str
        Path to the npy vector containing order of presented motion directions.
    vecVslStim: list
        Key of (stimulus) condition presented in every run
    Returns
    -------
    aryPresOrdAprt : 1d numpy array, shape [n_vols]
        Presentation order of aperture position.
    aryPresOrdMtn : 1d numpy array, shape [n_vols]
        Presentation order of motion direction.
    """
    print('------Load presentation order of motion directions')
    aryPresOrd = np.empty((0, 2))
    for idx01 in range(0, len(vecRunLngth)):
        # reconstruct file name
        # ---> consider: some runs were shorter than others(replace next row)
        filename1 = (strPathPresOrd + str(vecVslStim[idx01]) +
                     '.pickle')
        # filename1 = (strPathPresOrd + str(idx01+1) + '.pickle')
        # load array
        with open(filename1, 'rb') as handle:
            array1 = pickle.load(handle)
        tempCond = array1["Conditions"]
        tempCond = tempCond[:vecRunLngth[idx01], :]
        # add temp array to aryPresOrd
        aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
    aryPresOrdAprt = aryPresOrd[:, 0].astype(int)
    aryPresOrdMtn = aryPresOrd[:, 1].astype(int)

    return aryPresOrdAprt, aryPresOrdMtn