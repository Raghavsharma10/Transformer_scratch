def _slice_idxs(df, twin=None):
    """
    Returns a slice of the incoming array filtered between
    the two times specified. Assumes the array is the same
    length as self.data. Acts in the time() and trace() functions.
    """
    if twin is None:
        return 0, df.shape[0]

    tme = df.index

    if twin[0] is None:
        st_idx = 0
    else:
        st_idx = (np.abs(tme - twin[0])).argmin()
    if twin[1] is None:
        en_idx = df.shape[0]
    else:
        en_idx = (np.abs(tme - twin[1])).argmin() + 1
    return st_idx, en_idx