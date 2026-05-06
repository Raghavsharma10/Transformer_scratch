def score_at_fpr(fg_vals, bg_vals, fpr=0.01):
    """
    Returns the motif score at a specific FPR (default 1%).

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.
    
    fpr : float, optional
        The FPR (between 0.0 and 1.0).
    
    Returns
    -------
    score : float
        The motif score at the specified FPR.
    """
    bg_vals = np.array(bg_vals)
    return scoreatpercentile(bg_vals, 100 - 100 * fpr)