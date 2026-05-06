def Beta(alpha, beta, low=0, high=1, tag=None):
    """
    A Beta random variate
    
    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter
    
    Optional
    --------
    low : scalar
        Lower bound of the distribution support (default=0)
    high : scalar
        Upper bound of the distribution support (default=1)
    """
    assert (
        alpha > 0 and beta > 0
    ), 'Beta "alpha" and "beta" parameters must be greater than zero'
    assert low < high, 'Beta "low" must be less than "high"'
    return uv(ss.beta(alpha, beta, loc=low, scale=high - low), tag=tag)