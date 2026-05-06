def mtanh(alpha, z):
    """Modified hyperbolic tangent function mtanh(z; alpha).
    
    Parameters
    ----------
    alpha : float
        The core slope of the mtanh.
    z : float or array
        The coordinate of the mtanh.
    """
    z = scipy.asarray(z)
    ez = scipy.exp(z)
    enz = 1.0 / ez
    return ((1 + alpha * z) * ez - enz) / (ez + enz)