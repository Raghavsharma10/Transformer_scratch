def ks_pvalue(fg_pos, bg_pos=None):
    """
    Computes the Kolmogorov-Smirnov p-value of position distribution.

    Parameters
    ----------
    fg_pos : array_like
        The list of values for the positive set.

    bg_pos : array_like, optional
        The list of values for the negative set.
    
    Returns
    -------
    p : float
        KS p-value.
    """
    if len(fg_pos) == 0:
        return 1.0
    a = np.array(fg_pos, dtype="float") / max(fg_pos)
    p = kstest(a, "uniform")[1]
    return p