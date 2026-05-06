def Normal(mu, sigma, tag=None):
    """
    A Normal (or Gaussian) random variate
    
    Parameters
    ----------
    mu : scalar
        The mean value of the distribution
    sigma : scalar
        The standard deviation (must be positive and non-zero)
    """
    assert sigma > 0, 'Normal "sigma" must be greater than zero'
    return uv(ss.norm(loc=mu, scale=sigma), tag=tag)