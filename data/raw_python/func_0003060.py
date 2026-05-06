def joinRes(lstPrfRes, varPar, idxPos, inFormat='1D'):
    """Join results from different processing units (here cores).

    Parameters
    ----------
    lstPrfRes : list
        Output of results from parallelization.
    varPar : integer, positive
        Number of cores that were used during parallelization
    idxPos : integer, positive
        List position index that we expect the results to be collected to have.
    inFormat : string
        Specifies whether input will be 1d or 2d.

    Returns
    -------
    aryOut : numpy array
        Numpy array with results collected from different cores

    """

    if inFormat == '1D':
        # initialize output array
        aryOut = np.zeros((0,))
        # gather arrays from different processing units
        for idxRes in range(0, varPar):
            aryOut = np.append(aryOut, lstPrfRes[idxRes][idxPos])

    elif inFormat == '2D':
        # initialize output array
        aryOut = np.zeros((0, lstPrfRes[0][idxPos].shape[-1]))
        # gather arrays from different processing units
        for idxRes in range(0, varPar):
            aryOut = np.concatenate((aryOut, lstPrfRes[idxRes][idxPos]),
                                    axis=0)

    return aryOut