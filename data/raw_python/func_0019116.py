def hsepd(sim=None, obs=None, node=None, skip_nan=False,
          inits=None, return_pars=False, silent=True):
    """Calculate the mean of the logarithmised probability densities of the
    'heteroskedastic skewed exponential power distribution.

    Function |hsepd| serves the same purpose as function |hsepd_manual|,
    but tries to estimate the parameters of the heteroscedastic skewed
    exponential distribution via an optimization algorithm.  This
    is shown by generating a random sample.  1000 simulated values
    are scattered around the observed (true) value of 10.0 with a
    standard deviation of 2.0:

    >>> import numpy
    >>> numpy.random.seed(0)
    >>> sim = numpy.random.normal(10.0, 2.0, 1000)
    >>> obs = numpy.full(1000, 10.0)

    First, as a reference, we calculate the "true" value based on
    function |hsepd_manual| and the correct distribution parameters:

    >>> from hydpy import round_
    >>> from hydpy import hsepd, hsepd_manual
    >>> round_(hsepd_manual(sigma1=0.2, sigma2=0.0,
    ...                     xi=1.0, beta=0.0,
    ...                     sim=sim, obs=obs))
    -2.100093

    When using function |hsepd|, the returned value is even a little
    "better":

    >>> round_(hsepd(sim=sim, obs=obs))
    -2.09983

    This is due to the deviation from the random sample to its
    theoretical distribution.  This is reflected by small differences
    between the estimated values and the theoretical values of
    `sigma1` (0.2), , `sigma2` (0.0), `xi` (1.0), and `beta` (0.0).
    The estimated values are returned in the mentioned order through
    enabling the `return_pars` option:

    >>> value, pars = hsepd(sim=sim, obs=obs, return_pars=True)
    >>> round_(pars, decimals=5)
    0.19966, 0.0, 0.96836, 0.0188

    There is no guarantee that the optimization numerical optimization
    algorithm underlying function |hsepd| will always find the parameters
    resulting in the largest value returned by function |hsepd_manual|.
    You can increase its robustness (and decrease computation time) by
    supplying good initial parameter values:

    >>> value, pars = hsepd(sim=sim, obs=obs, return_pars=True,
    ...                     inits=(0.2, 0.0, 1.0, 0.0))
    >>> round_(pars, decimals=5)
    0.19966, 0.0, 0.96836, 0.0188

    However, the following example shows a case when this strategie
    results in worse results:

    >>> value, pars = hsepd(sim=sim, obs=obs, return_pars=True,
    ...                     inits=(0.0, 0.2, 1.0, 0.0))
    >>> round_(value)
    -2.174492
    >>> round_(pars)
    0.0, 0.213179, 1.705485, 0.505112
    """

    def transform(pars):
        """Transform the actual optimization problem into a function to
        be minimized and apply parameter constraints."""
        sigma1, sigma2, xi, beta = constrain(*pars)
        return -_hsepd_manual(sigma1, sigma2, xi, beta, sim, obs)

    def constrain(sigma1, sigma2, xi, beta):
        """Apply constrains on the given parameter values."""
        sigma1 = numpy.clip(sigma1, 0.0, None)
        sigma2 = numpy.clip(sigma2, 0.0, None)
        xi = numpy.clip(xi, 0.1, 10.0)
        beta = numpy.clip(beta, -0.99, 5.0)
        return sigma1, sigma2, xi, beta

    sim, obs = prepare_arrays(sim, obs, node, skip_nan)
    if not inits:
        inits = [0.1, 0.2, 3.0, 1.0]
    values = optimize.fmin(transform, inits,
                           ftol=1e-12, xtol=1e-12,
                           disp=not silent)
    values = constrain(*values)
    result = _hsepd_manual(*values, sim=sim, obs=obs)
    if return_pars:
        return result, values
    return result