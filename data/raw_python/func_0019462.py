def fraction_fpr(fg_vals, bg_vals, fpr=0.01):
    """
    Computes the fraction positives at a specific FPR (default 1%).

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
    fraction : float
        The fraction positives at the specified FPR.
    """
    fg_vals = np.array(fg_vals)
    s = scoreatpercentile(bg_vals, 100 - 100 * fpr)
    return len(fg_vals[fg_vals >= s]) / float(len(fg_vals))