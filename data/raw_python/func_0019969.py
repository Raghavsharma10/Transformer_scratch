def Poisson(lamda, tag=None):
    """
    A Poisson random variate
    
    Parameters
    ----------
    lamda : scalar
        The rate of an occurance within a specified interval of time or space.
    """
    assert lamda > 0, 'Poisson "lamda" must be greater than zero.'
    return uv(ss.poisson(lamda), tag=tag)