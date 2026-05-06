def BetaPrime(alpha, beta, tag=None):
    """
    A BetaPrime random variate
    
    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter
    
    """
    assert (
        alpha > 0 and beta > 0
    ), 'BetaPrime "alpha" and "beta" parameters must be greater than zero'
    x = Beta(alpha, beta, tag)
    return x / (1 - x)