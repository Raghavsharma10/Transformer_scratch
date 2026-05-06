def Bernoulli(p, tag=None):
    """
    A Bernoulli random variate
    
    Parameters
    ----------
    p : scalar
        The probability of success
    """
    assert (
        0 < p < 1
    ), 'Bernoulli probability "p" must be between zero and one, non-inclusive'
    return uv(ss.bernoulli(p), tag=tag)