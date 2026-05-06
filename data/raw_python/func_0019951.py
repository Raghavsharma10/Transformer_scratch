def Erlang(k, lamda, tag=None):
    """
    An Erlang random variate.
    
    This distribution is the same as a Gamma(k, theta) distribution, but 
    with the restriction that k must be a positive integer. This
    is provided for greater compatibility with other simulation tools, but
    provides no advantage over the Gamma distribution in its applications.
    
    Parameters
    ----------
    k : int
        The shape parameter (must be a positive integer)
    lamda : scalar
        The scale parameter (must be greater than zero)
    """
    assert int(k) == k and k > 0, 'Erlang "k" must be a positive integer'
    assert lamda > 0, 'Erlang "lamda" must be greater than zero'
    return Gamma(k, lamda, tag)