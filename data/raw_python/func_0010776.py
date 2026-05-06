def linregress(x, y, return_stats=False):
    """linear regression calculation

    Parameters
    ----
    x :         independent variable (series)
    y :         dependent variable (series)
    return_stats : returns statistical values as well if required (bool)
    

    Returns
    ----
    list of parameters (and statistics)
    """
    a1, a0, r_value, p_value, stderr = scipy.stats.linregress(x, y)

    retval = a1, a0
    if return_stats:
        retval += r_value, p_value, stderr

    return retval