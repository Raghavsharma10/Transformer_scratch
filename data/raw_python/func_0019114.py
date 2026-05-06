def hsepd_pdf(sigma1, sigma2, xi, beta,
              sim=None, obs=None, node=None, skip_nan=False):
    """Calculate the probability densities based on the
    heteroskedastic skewed exponential power distribution.

    For convenience, the required parameters of the probability density
    function as well as the simulated and observed values are stored
    in a dictonary:

    >>> import numpy
    >>> from hydpy import round_
    >>> from hydpy import hsepd_pdf
    >>> general = {'sigma1': 0.2,
    ...            'sigma2': 0.0,
    ...            'xi': 1.0,
    ...            'beta': 0.0,
    ...            'sim': numpy.arange(10.0, 41.0),
    ...            'obs': numpy.full(31, 25.0)}

    The following test function allows the variation of one parameter
    and prints some and plots all of probability density values
    corresponding to different simulated values:

    >>> def test(**kwargs):
    ...     from matplotlib import pyplot
    ...     special = general.copy()
    ...     name, values = list(kwargs.items())[0]
    ...     results = numpy.zeros((len(general['sim']), len(values)+1))
    ...     results[:, 0] = general['sim']
    ...     for jdx, value in enumerate(values):
    ...         special[name] = value
    ...         results[:, jdx+1] = hsepd_pdf(**special)
    ...         pyplot.plot(results[:, 0], results[:, jdx+1],
    ...                     label='%s=%.1f' % (name, value))
    ...     pyplot.legend()
    ...     for idx, result in enumerate(results):
    ...         if not (idx % 5):
    ...             round_(result)

    When varying parameter `beta`, the resulting probabilities correspond
    to the Laplace distribution (1.0), normal distribution (0.0), and the
    uniform distribution (-1.0), respectively.  Note that we use -0.99
    instead of -1.0 for approximating the uniform distribution to prevent
    from running into numerical problems, which are not solved yet:

    >>> test(beta=[1.0, 0.0, -0.99])
    10.0, 0.002032, 0.000886, 0.0
    15.0, 0.008359, 0.010798, 0.0
    20.0, 0.034382, 0.048394, 0.057739
    25.0, 0.141421, 0.079788, 0.057739
    30.0, 0.034382, 0.048394, 0.057739
    35.0, 0.008359, 0.010798, 0.0
    40.0, 0.002032, 0.000886, 0.0

    .. testsetup::

        >>> from matplotlib import pyplot
        >>> pyplot.close()

    When varying parameter `xi`, the resulting density is negatively
    skewed (0.2), symmetric (1.0), and positively skewed (5.0),
    respectively:

    >>> test(xi=[0.2, 1.0, 5.0])
    10.0, 0.0, 0.000886, 0.003175
    15.0, 0.0, 0.010798, 0.012957
    20.0, 0.092845, 0.048394, 0.036341
    25.0, 0.070063, 0.079788, 0.070063
    30.0, 0.036341, 0.048394, 0.092845
    35.0, 0.012957, 0.010798, 0.0
    40.0, 0.003175, 0.000886, 0.0

    .. testsetup::

        >>> from matplotlib import pyplot
        >>> pyplot.close()

    In the above examples, the actual `sigma` (5.0) is calculated by
    multiplying `sigma1` (0.2) with the mean simulated value (25.0),
    internally.  This can be done for modelling homoscedastic errors.
    Instead, `sigma2` is multiplied with the individual simulated values
    to account for heteroscedastic errors.  With increasing values of
    `sigma2`, the resulting densities are modified as follows:

    >>> test(sigma2=[0.0, 0.1, 0.2])
    10.0, 0.000886, 0.002921, 0.005737
    15.0, 0.010798, 0.018795, 0.022831
    20.0, 0.048394, 0.044159, 0.037988
    25.0, 0.079788, 0.053192, 0.039894
    30.0, 0.048394, 0.04102, 0.032708
    35.0, 0.010798, 0.023493, 0.023493
    40.0, 0.000886, 0.011053, 0.015771

    .. testsetup::

        >>> from matplotlib import pyplot
        >>> pyplot.close()
    """
    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    sigmas = _pars_h(sigma1, sigma2, sim)
    mu_xi, sigma_xi, w_beta, c_beta = _pars_sepd(xi, beta)
    x, mu = obs, sim
    a = (x-mu)/sigmas
    a_xi = numpy.empty(a.shape)
    idxs = mu_xi+sigma_xi*a < 0.
    a_xi[idxs] = numpy.absolute(xi*(mu_xi+sigma_xi*a[idxs]))
    a_xi[~idxs] = numpy.absolute(1./xi*(mu_xi+sigma_xi*a[~idxs]))
    ps = (2.*sigma_xi/(xi+1./xi)*w_beta *
          numpy.exp(-c_beta*a_xi**(2./(1.+beta))))/sigmas
    return ps