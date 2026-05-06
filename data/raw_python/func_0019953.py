def ExtValueMax(mu, sigma, tag=None):
    """
    An Extreme Value Maximum random variate.
    
    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be greater than zero)
    """
    assert sigma > 0, 'ExtremeValueMax "sigma" must be greater than zero'
    p = U(0, 1)._mcpts[:]
    return UncertainFunction(mu - sigma * np.log(-np.log(p)), tag=tag)