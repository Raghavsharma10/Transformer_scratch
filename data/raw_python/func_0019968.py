def Hypergeometric(N, n, K, tag=None):
    """
    A Hypergeometric random variate
    
    Parameters
    ----------
    N : int
        The total population size
    n : int
        The number of individuals of interest in the population
    K : int
        The number of individuals that will be chosen from the population
        
    Example
    -------
    (Taken from the wikipedia page) Assume we have an urn with two types of
    marbles, 45 black ones and 5 white ones. Standing next to the urn, you
    close your eyes and draw 10 marbles without replacement. What is the
    probability that exactly 4 of the 10 are white?
    ::
    
        >>> black = 45
        >>> white = 5
        >>> draw = 10
        
        # Now we create the distribution
        >>> h = H(black + white, white, draw)
        
        # To check the probability, in this case, we can use the underlying
        #  scipy.stats object
        >>> h.rv.pmf(4)  # What is the probability that white count = 4?
        0.0039645830580151975
        
    """
    assert (
        int(N) == N and N > 0
    ), 'Hypergeometric total population size "N" must be an integer greater than zero.'
    assert (
        int(n) == n and 0 < n <= N
    ), 'Hypergeometric interest population size "n" must be an integer greater than zero and no more than the total population size.'
    assert (
        int(K) == K and 0 < K <= N
    ), 'Hypergeometric chosen population size "K" must be an integer greater than zero and no more than the total population size.'
    return uv(ss.hypergeom(N, n, K), tag=tag)