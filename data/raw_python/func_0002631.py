def load_ev_txt(strPthEv):
    """Load information from event text file.

    Parameters
    ----------
    input1 : str
        Path to event text file
    Returns
    -------
    aryEvTxt : 2d numpy array, shape [n_measurements, 3]
        Array with info about conditions: type, onset, duration
    Notes
    -----
    Part of py_pRF_mapping library.
    """
    aryEvTxt = np.loadtxt(strPthEv, dtype='float', comments='#', delimiter=' ',
                          skiprows=0, usecols=(0, 1, 2))
    return aryEvTxt