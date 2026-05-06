def create_boxcar(aryCnd, aryOns, aryDrt, varTr, varNumVol,
                  aryExclCnd=None, varTmpOvsmpl=1000.):
    """
    Creation of condition time courses in temporally upsampled space.

    Parameters
    ----------

    aryCnd : np.array
        1D array with condition identifiers (every condition has its own int)
    aryOns : np.array, same len as aryCnd
        1D array with condition onset times in seconds.
    aryDrt : np.array, same len as aryCnd
        1D array with condition durations of different conditions in seconds.
    varTr : float, positive
        Time to repeat (TR) of the (fMRI) experiment.
    varNumVol : float, positive
        Number of volumes of the (fMRI) data.
    aryExclCnd : array
        1D array containing condition identifiers for conditions to be excluded
    varTmpOvsmpl : float, positive
        Factor by which the time courses should be temporally upsampled.

    Returns
    -------
    aryBxCrOut : np.array, float16
        Condition time courses in temporally upsampled space.

    References:
    -----
    [1] https://github.com/fabianp/hrf_estimation

    """
    if aryExclCnd is not None:
        for cond in aryExclCnd:
            aryOns = aryOns[aryCnd != cond]
            aryDrt = aryDrt[aryCnd != cond]
            aryCnd = aryCnd[aryCnd != cond]

    resolution = varTr / float(varTmpOvsmpl)
    aryCnd = np.asarray(aryCnd)
    aryOns = np.asarray(aryOns, dtype=np.float)
    unique_conditions = np.sort(np.unique(aryCnd))
    boxcar = []

    for c in unique_conditions:
        tmp = np.zeros(int(varNumVol * varTr/resolution))
        onset_c = aryOns[aryCnd == c]
        duration_c = aryDrt[aryCnd == c]
        onset_idx = np.round(onset_c / resolution).astype(np.int)
        duration_idx = np.round(duration_c / resolution).astype(np.int)
        aux = np.arange(int(varNumVol * varTr/resolution))
        for start, dur in zip(onset_idx, duration_idx):
            lgc = np.logical_and(aux >= start, aux < start + dur)
            tmp = tmp + lgc
        assert np.all(np.less(tmp, 2))
        boxcar.append(tmp)
    aryBxCrOut = np.array(boxcar).T
    if aryBxCrOut.shape[1] == 1:
        aryBxCrOut = np.squeeze(aryBxCrOut)
    return aryBxCrOut.astype('float16')