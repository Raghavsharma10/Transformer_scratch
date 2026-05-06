def max_enrichment(fg_vals, bg_vals, minbg=2):
    """
    Computes the maximum enrichment.

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.
    
    minbg : int, optional
        Minimum number of matches in background. The default is 2.
    
    Returns
    -------
    enrichment : float
        Maximum enrichment.
    """
    scores = np.hstack((fg_vals, bg_vals))
    idx = np.argsort(scores)
    x = np.hstack((np.ones(len(fg_vals)), np.zeros(len(bg_vals))))
    xsort = x[idx]
    l_fg = len(fg_vals)
    l_bg = len(bg_vals)
    m = 0
    s = 0
    for i in range(len(scores), 0, -1):
        bgcount = float(len(xsort[i:][xsort[i:] == 0])) 
        if bgcount >= minbg:
            enr = (len(xsort[i:][xsort[i:] == 1]) / l_fg) / (bgcount / l_bg)
            if enr > m:
                m = enr
                s = scores[idx[i]]
    return m