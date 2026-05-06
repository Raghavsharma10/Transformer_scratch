def beta_cdf_warp(X, d, n, *args):
    r"""Warp inputs that are confined to the unit hypercube using the regularized incomplete beta function.
    
    Applies separately to each dimension, designed for use with
    :py:class:`WarpingFunction`.
    
    Assumes that your inputs `X` lie entirely within the unit hypercube [0, 1].
    
    Note that you may experience some issues with constraining and computing
    derivatives at :math:`x=0` when :math:`\alpha < 1` and at :math:`x=1` when
    :math:`\beta < 1`. As a workaround, try mapping your data to not touch the
    boundaries of the unit hypercube.
    
    Parameters
    ----------
    X : array, (`M`,)
        `M` inputs from dimension `d`.
    d : non-negative int
        The index (starting from zero) of the dimension to apply the warping to.
    n : non-negative int
        The derivative order to compute.
    *args : 2N scalars
        The remaining parameters to describe the warping, given as scalars.
        These are given as `alpha_i`, `beta_i` for each of the `D` dimensions.
        Note that these must ALL be provided for each call.
    
    References
    ----------
    .. [1] J. Snoek, K. Swersky, R. Zemel, R. P. Adams, "Input Warping for
       Bayesian Optimization of Non-stationary Functions" ICML (2014)
    """
    X = scipy.asarray(X)
    
    a = args[2 * d]
    b = args[2 * d + 1]
    
    if n == 0:
        return scipy.special.betainc(a, b, X)
    elif n == 1:
        # http://functions.wolfram.com/GammaBetaErf/BetaRegularized/20/01/01/
        return (1 - X)**(b - 1) * X**(a - 1) / scipy.special.beta(a, b)
    else:
        # http://functions.wolfram.com/GammaBetaErf/BetaRegularized/20/02/01/
        out = scipy.zeros_like(X)
        for k in range(0, n):
            out += (
                (-1.0)**(n - k) * scipy.special.binom(n - 1, k) *
                fixed_poch(1.0 - b, k) * fixed_poch(1.0 - a, n - k - 1.0) *
                (X / (1.0 - X))**k
            )
        return -(1.0 - X)**(b - 1.0) * X**(a - n) * out / scipy.special.beta(a, b)