def smooth_differentiation(x, y, weigths=None, order=5, smoothness=3, derivation=1):
    '''Returns the dy/dx(x) with the fit and differentiation of a spline curve

    Parameters
    ----------
    x : array like
    y : array like

    Returns
    -------
    dy/dx : array like
    '''
    if (len(x) != len(y)):
        raise ValueError("x, y must have the same length")
    f = splrep(x, y, w=weigths, k=order, s=smoothness)  # spline function
    return splev(x, f, der=derivation)