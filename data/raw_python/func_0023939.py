def exp_gauss_warp(X, n, l0, *msb):
    """Length scale function which is an exponential of a sum of Gaussians.
    
    The centers and widths of the Gaussians are free parameters.
    
    The length scale function is given by
    
    .. math::
        
        l = l_0 \exp\left ( \sum_{i=1}^{N}\beta_i\exp\left ( -\frac{(x-\mu_i)^2}{2\sigma_i^2} \right ) \right )
    
    The number of parameters is equal to the three times the number of Gaussians
    plus 1 (for :math:`l_0`). This function is inspired by what Gibbs used in
    his PhD thesis.
    
    Parameters
    ----------
    X : 1d or 2d array of float
        The points to evaluate the function at. If 2d, it should only have
        one column (but this is not checked to save time).
    n : int
        The derivative order to compute. Used for all `X`.
    l0 : float
        The covariance length scale at the edges of the domain.
    *msb : floats
        Means, standard deviations and weights for each Gaussian, in that order.
    """
    X = scipy.asarray(X, dtype=float)
    msb = scipy.asarray(msb, dtype=float)
    mm = msb[:len(msb) / 3]
    ss = msb[len(msb) / 3:2 * len(msb) / 3]
    bb = msb[2 * len(msb) / 3:]
    
    # This is done with for-loops, because trying to get fancy with
    # broadcasting was being too memory-intensive for some reason.
    if n == 0:
        l = scipy.zeros_like(X)
        for m, s, b in zip(mm, ss, bb):
            l += b * scipy.exp(-(X - m)**2.0 / (2.0 * s**2.0))
        l = l0 * scipy.exp(l)
        return l
    elif n == 1:
        l1 = scipy.zeros_like(X)
        l2 = scipy.zeros_like(X)
        for m, s, b in zip(mm, ss, bb):
            term = b * scipy.exp(-(X - m)**2.0 / (2.0 * s**2.0))
            l1 += term
            l2 += term * (X - m) / s**2.0
        l = -l0 * scipy.exp(l1) * l2
        return l
    else:
        raise NotImplementedError("Only n <= 1 is supported!")