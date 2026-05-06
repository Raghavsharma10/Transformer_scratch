def hsepd_manual(sigma1, sigma2, xi, beta,
                 sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the mean of the logarithmised probability densities of the
    'heteroskedastic skewed exponential power distribution.

    The following examples are taken from the documentation of function
    |hsepd_pdf|, which is used by function |hsepd_manual|.  The first
    one deals with a heteroscedastic normal distribution:

    >>> from hydpy import round_
    >>> from hydpy import hsepd_manual
    >>> round_(hsepd_manual(sigma1=0.2, sigma2=0.2,
    ...                     xi=1.0, beta=0.0,
    ...                     sim=numpy.arange(10.0, 41.0),
    ...                     obs=numpy.full(31, 25.0)))
    -3.682842

    The second one is supposed to show to small zero probability density
    values are set to 1e-200 before calculating their logarithm (which
    means that the lowest possible value returned by function
    |hsepd_manual| is approximately -460):

    >>> round_(hsepd_manual(sigma1=0.2, sigma2=0.0,
    ...                     xi=1.0, beta=-0.99,
    ...                     sim=numpy.arange(10.0, 41.0),
    ...                     obs=numpy.full(31, 25.0)))
    -209.539335
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    return _hsepd_manual(sigma1, sigma2, xi, beta, sim, obs)