def Pareto2(q, b, tag=None):
    """
    A Pareto random variate (second kind). This form always starts at the
    origin.
    
    Parameters
    ----------
    q : scalar
        The scale parameter
    b : scalar
        The shape parameter
    """
    assert q > 0 and b > 0, 'Pareto2 "q" and "b" must be positive scalars'
    return Pareto(q, b, tag) - b