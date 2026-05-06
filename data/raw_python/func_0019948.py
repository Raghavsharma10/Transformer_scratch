def Bradford(q, low=0, high=1, tag=None):
    """
    A Bradford random variate
    
    Parameters
    ----------
    q : scalar
        The shape parameter
    low : scalar
        The lower bound of the distribution (default=0)
    high : scalar
        The upper bound of the distribution (default=1)
    """
    assert q > 0, 'Bradford "q" parameter must be greater than zero'
    assert low < high, 'Bradford "low" parameter must be less than "high"'
    return uv(ss.bradford(q, loc=low, scale=high - low), tag=tag)