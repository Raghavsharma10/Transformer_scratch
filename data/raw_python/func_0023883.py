def matern_function(Xi, Xj, *args):
    r"""Matern covariance function of arbitrary dimension, for use with :py:class:`ArbitraryKernel`.
    
    The Matern kernel has the following hyperparameters, always referenced in
    the order listed:
    
    = ===== ====================================
    0 sigma prefactor
    1 nu    order of kernel
    2 l1    length scale for the first dimension
    3 l2    ...and so on for all dimensions
    = ===== ====================================
    
    The kernel is defined as:
    
    .. math::
    
        k_M = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
        \left (\sqrt{2\nu \sum_i\left (\frac{\tau_i^2}{l_i^2}\right )}\right )^\nu
        K_\nu\left(\sqrt{2\nu \sum_i\left(\frac{\tau_i^2}{l_i^2}\right)}\right)
    
    Parameters
    ----------
    Xi, Xj : :py:class:`Array`, :py:class:`mpf`, tuple or scalar float
        Points to evaluate the covariance between. If they are :py:class:`Array`,
        :py:mod:`scipy` functions are used, otherwise :py:mod:`mpmath`
        functions are used.
    *args
        Remaining arguments are the 2+num_dim hyperparameters as defined above.
    """
    num_dim = len(args) - 2
    nu = args[1]
    
    if isinstance(Xi, scipy.ndarray):
        if isinstance(Xi, scipy.matrix):
            Xi = scipy.asarray(Xi, dtype=float)
            Xj = scipy.asarray(Xj, dtype=float)
        
        tau = scipy.asarray(Xi - Xj, dtype=float)
        l_mat = scipy.tile(args[-num_dim:], (tau.shape[0], 1))
        r2l2 = scipy.sum((tau / l_mat)**2, axis=1)
        y = scipy.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / scipy.special.gamma(nu) * y**nu * scipy.special.kv(nu, y)
        k[r2l2 == 0] = 1
    else:
        try:
            tau = [xi - xj for xi, xj in zip(Xi, Xj)]
        except TypeError:
            tau = Xi - Xj
        try:
            r2l2 = sum([(t / l)**2 for t, l in zip(tau, args[2:])])
        except TypeError:
            r2l2 = (tau / args[2])**2
        y = mpmath.sqrt(2.0 * nu * r2l2)
        k = 2.0**(1 - nu) / mpmath.gamma(nu) * y**nu * mpmath.besselk(nu, y)
    k *= args[0]**2.0
    return k