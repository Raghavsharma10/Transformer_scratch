def mncp(fg_vals, bg_vals):
    """
    Computes the Mean Normalized Conditional Probability (MNCP).

    MNCP is described in Clarke & Granek, Bioinformatics, 2003.

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.
    
    Returns
    -------
    score : float
        MNCP score
    """
    fg_len = len(fg_vals)
    total_len = len(fg_vals) + len(bg_vals)

    if not isinstance(fg_vals, np.ndarray):
        fg_vals = np.array(fg_vals)
    if not isinstance(bg_vals, np.ndarray):
        bg_vals = np.array(bg_vals)
    
    fg_rank = stats.rankdata(fg_vals)
    total_rank = stats.rankdata(np.hstack((fg_vals, bg_vals)))

    slopes = []
    for i in range(len(fg_vals)):
        slope = ((fg_len - fg_rank[i] + 1) / fg_len ) / (
                (total_len - total_rank[i] + 1)/ total_len)
        slopes.append(slope)
    
    return np.mean(slopes)