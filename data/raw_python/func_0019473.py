def ks_significance(fg_pos, bg_pos=None):
    """
    Computes the -log10 of Kolmogorov-Smirnov p-value of position distribution.

    Parameters
    ----------
    fg_pos : array_like
        The list of values for the positive set.

    bg_pos : array_like, optional
        The list of values for the negative set.
    
    Returns
    -------
    p : float
        -log10(KS p-value).
    """
    p = ks_pvalue(fg_pos, max(fg_pos))
    if p > 0:
        return -np.log10(p)
    else:
        return np.inf