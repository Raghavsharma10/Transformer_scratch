def Burr(c, k, tag=None):
    """
    A Burr random variate
    
    Parameters
    ----------
    c : scalar
        The first shape parameter
    k : scalar
        The second shape parameter
    
    """
    assert c > 0 and k > 0, 'Burr "c" and "k" parameters must be greater than zero'
    return uv(ss.burr(c, k), tag=tag)