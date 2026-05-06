def Exponential(lamda, tag=None):
    """
    An Exponential random variate
    
    Parameters
    ----------
    lamda : scalar
        The inverse scale (as shown on Wikipedia). (FYI: mu = 1/lamda.)
    """
    assert lamda > 0, 'Exponential "lamda" must be greater than zero'
    return uv(ss.expon(scale=1.0 / lamda), tag=tag)