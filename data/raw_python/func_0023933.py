def tanh_warp_arb(X, l1, l2, lw, x0):
    r"""Warps the `X` coordinate with the tanh model
    
    .. math::
    
        l = \frac{l_1 + l_2}{2} - \frac{l_1 - l_2}{2}\tanh\frac{x-x_0}{l_w}
    
    Parameters
    ----------
    X : :py:class:`Array`, (`M`,) or scalar float
        `M` locations to evaluate length scale at.
    l1 : positive float
        Small-`X` saturation value of the length scale.
    l2 : positive float
        Large-`X` saturation value of the length scale.
    lw : positive float
        Length scale of the transition between the two length scales.
    x0 : float
        Location of the center of the transition between the two length scales.
    
    Returns
    -------
    l : :py:class:`Array`, (`M`,) or scalar float
        The value of the length scale at the specified point.
    """
    if isinstance(X, scipy.ndarray):
        if isinstance(X, scipy.matrix):
            X = scipy.asarray(X, dtype=float)
        return 0.5 * ((l1 + l2) - (l1 - l2) * scipy.tanh((X - x0) / lw))
    else:
        return 0.5 * ((l1 + l2) - (l1 - l2) * mpmath.tanh((X - x0) / lw))