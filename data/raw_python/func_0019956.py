def LogNormal(mu, sigma, tag=None):
    """
    A Log-Normal random variate
    
    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be positive and non-zero)
    """
    assert sigma > 0, 'Log-Normal "sigma" must be positive'
    return uv(ss.lognorm(sigma, loc=mu), tag=tag)