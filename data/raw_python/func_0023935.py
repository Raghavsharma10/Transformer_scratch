def tanh_warp(x, n, l1, l2, lw, x0):
    r"""Implements a tanh warping function and its derivative.
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    l1 : positive float
        Left saturation value.
    l2 : positive float
        Right saturation value.
    lw : positive float
        Transition width.
    x0 : float
        Transition location.
    
    Returns
    -------
    l : float or array
        Warped length scale at the given locations.
    
    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    if n == 0:
        return (l1 + l2) / 2.0 - (l1 - l2) / 2.0 * scipy.tanh((x - x0) / lw)
    elif n == 1:
        return -(l1 - l2) / (2.0 * lw) * (scipy.cosh((x - x0) / lw))**(-2.0)
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")