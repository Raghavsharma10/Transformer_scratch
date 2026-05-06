def Binomial(n, p, tag=None):
    """
    A Binomial random variate
    
    Parameters
    ----------
    n : int
        The number of trials
    p : scalar
        The probability of success
    """
    assert (
        int(n) == n and n > 0
    ), 'Binomial number of trials "n" must be an integer greater than zero'
    assert (
        0 < p < 1
    ), 'Binomial probability "p" must be between zero and one, non-inclusive'
    return uv(ss.binom(n, p), tag=tag)