def Weibull(lamda, k, tag=None):
    """
    A Weibull random variate
    
    Parameters
    ----------
    lamda : scalar
        The scale parameter
    k : scalar
        The shape parameter
    """
    assert (
        lamda > 0 and k > 0
    ), 'Weibull "lamda" and "k" parameters must be greater than zero'
    return uv(ss.exponweib(lamda, k), tag=tag)