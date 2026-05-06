def max_fmeasure(fg_vals, bg_vals):
    """
    Computes the maximum F-measure.

    Parameters
    ----------
    fg_vals : array_like
        The list of values for the positive set.

    bg_vals : array_like
        The list of values for the negative set.
    
    Returns
    -------
    f : float
        Maximum f-measure.
    """
    x, y = roc_values(fg_vals, bg_vals)
    x, y = x[1:], y[1:] # don't include origin
    
    p = y / (y + x)
    filt = np.logical_and((p * y) > 0, (p + y) > 0)
    p = p[filt]
    y = y[filt]
    
    f = (2 * p * y) / (p + y)
    if len(f) > 0:
        #return np.nanmax(f), np.nanmax(y[f == np.nanmax(f)])
        return np.nanmax(f)
    else:
        return None