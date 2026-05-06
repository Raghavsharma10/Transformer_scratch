def gauss_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with a Gaussian-shaped divot.
    
    .. math::
        
        l = l_1 - (l_1 - l_2) \exp\left ( -4\ln 2\frac{(X-x_0)^2}{l_{w}^{2}} \right )
    
    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Global value of the length scale.
    l2 : positive float
        Pedestal value of the length scale.
    lw : positive float
        Width of the dip.
    x0 : float
        Location of the center of the dip in length scale.
    
    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return l1 - (l1 - l2) * scipy.exp(-4.0 * scipy.log(2.0) * (X - x0)**2.0 / (lw**2.0))
    else:
        return l1 - (l1 - l2) * mpmath.exp(-4.0 * mpmath.log(2.0) * (X - x0)**2.0 / (lw**2.0))