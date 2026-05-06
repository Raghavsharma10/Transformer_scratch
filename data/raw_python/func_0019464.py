def enr_at_fpr(fg_vals, bg_vals, fpr=0.01):
    """
    Computes the enrichment at a specific FPR (default 1%).

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
    enrichment : float
        The enrichment at the specified FPR.
    """
    pos = np.array(fg_vals)
    neg = np.array(bg_vals)
    s = scoreatpercentile(neg, 100 - fpr * 100)
    neg_matches = float(len(neg[neg >= s]))
    if neg_matches == 0:
        return float("inf")
    return len(pos[pos >= s]) / neg_matches * len(neg) / float(len(pos))