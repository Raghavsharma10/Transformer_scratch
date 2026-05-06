def Pareto(q, a, tag=None):
    """
    A Pareto random variate (first kind)
    
    Parameters
    ----------
    q : scalar
        The scale parameter
    a : scalar
        The shape parameter (the minimum possible value)
    """
    assert q > 0 and a > 0, 'Pareto "q" and "a" must be positive scalars'
    p = Uniform(0, 1, tag)
    return a * (1 - p) ** (-1.0 / q)