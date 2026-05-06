def Gamma(k, theta, tag=None):
    """
    A Gamma random variate
    
    Parameters
    ----------
    k : scalar
        The shape parameter (must be positive and non-zero)
    theta : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert (
        k > 0 and theta > 0
    ), 'Gamma "k" and "theta" parameters must be greater than zero'
    return uv(ss.gamma(k, scale=theta), tag=tag)