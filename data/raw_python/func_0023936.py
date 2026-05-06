def double_tanh_warp(x, n, lcore, lmid, ledge, la, lb, xa, xb):
    r"""Implements a sum-of-tanh warping function and its derivative.
    
    .. math::
    
        l = a\tanh\frac{x-x_a}{l_a} + b\tanh\frac{x-x_b}{l_b}
    
    Parameters
    ----------
    x : float or array of float
        Locations to evaluate the function at.
    n : int
        Derivative order to take. Used for ALL of the points.
    lcore : float
        Core length scale.
    lmid : float
        Intermediate length scale.
    ledge : float
        Edge length scale.
    la : positive float
        Transition of first tanh.
    lb : positive float
        Transition of second tanh.
    xa : float
        Transition of first tanh.
    xb : float
        Transition of second tanh.
    
    Returns
    -------
    l : float or array
        Warped length scale at the given locations.

    Raises
    ------
    NotImplementedError
        If `n` > 1.
    """
    a, b, c = scipy.dot([[-0.5, 0, 0.5], [0, 0.5, -0.5], [0.5, 0.5, 0]],
                        [[lcore], [ledge], [lmid]])
    a = a[0]
    b = b[0]
    c = c[0]
    if n == 0:
        return a * scipy.tanh((x - xa) / la) + b * scipy.tanh((x - xb) / lb) + c
    elif n == 1:
        return (a / la * (scipy.cosh((x - xa) / la))**(-2.0) +
                b / lb * (scipy.cosh((x - xb) / lb))**(-2.0))
    else:
        raise NotImplementedError("Only derivatives up to order 1 are supported!")