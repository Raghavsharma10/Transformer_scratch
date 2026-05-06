def Geometric(p, tag=None):
    """
    A Geometric random variate
    
    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert (
        0 < p < 1
    ), 'Geometric probability "p" must be between zero and one, non-inclusive'
    return uv(ss.geom(p), tag=tag)