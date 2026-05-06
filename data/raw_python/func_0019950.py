def ChiSquared(k, tag=None):
    """
    A Chi-Squared random variate
    
    Parameters
    ----------
    k : int
        The degrees of freedom of the distribution (must be greater than one)
    """
    assert int(k) == k and k >= 1, 'Chi-Squared "k" must be an integer greater than 0'
    return uv(ss.chi2(k), tag=tag)