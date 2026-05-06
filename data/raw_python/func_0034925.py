def calc_delta_c(c200):
    """Calculate characteristic overdensity from concentration.

    Parameters
    ----------
    c200 : ndarray or float
        Cluster concentration parameter.

    Returns
    ----------
    ndarray or float
        Cluster characteristic overdensity, of same type as c200.
    """
    top = (200. / 3.) * c200**3.
    bottom = np.log(1. + c200) - (c200 / (1. + c200))
    return (top / bottom)