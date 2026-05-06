def Uniform(low, high, tag=None):
    """
    A Uniform random variate
    
    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support.
    high : scalar
        Upper bound of the distribution support.
    """
    assert low < high, 'Uniform "low" must be less than "high"'
    return uv(ss.uniform(loc=low, scale=high - low), tag=tag)