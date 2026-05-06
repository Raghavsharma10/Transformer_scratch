def implementation_std(vals_std, vals_std_u, bs_std, bs_std_u, **kwargs):
    r"""Estimates varaition of results due to implementation-specific
    effects. See 'nestcheck: diagnostic tests for nested sampling calculations'
    (Higson et al. 2019) for more details.

    Uncertainties on the output are calculated numerically using the fact
    that (from central limit theorem) our uncertainties on vals_std and
    bs_std are (approximately) normally distributed. This is needed as
    results from standard error propagation techniques are not valid when
    the uncertainties are not small compared to the result.

    Parameters
    ----------
    vals_std: numpy array
        Standard deviations of results from repeated calculations.
    vals_std_u: numpy array
        :math:`1\sigma` uncertainties on vals_std_u.
    bs_std: numpy array
        Bootstrap error estimates. Each element should correspond to the same
        element in vals_std.
    bs_std_u: numpy array
        :math:`1\sigma` uncertainties on vals_std_u.
    nsim: int, optional
        Number of simulations to use to numerically calculate the uncertainties
        on the estimated implementation-specific effects.
    random_seed: int or None, optional
        Numpy random seed. Use to get reproducible uncertainties on the output.

    Returns
    -------
    imp_std: numpy array
        Estimated standard deviation of results due to implementation-specific
        effects.
    imp_std_u: numpy array
        :math:`1\sigma` uncertainties on imp_std.
    imp_frac: numpy array
        imp_std as a fraction of vals_std.
    imp_frac_u:
        :math:`1\sigma` uncertainties on imp_frac.
    """
    nsim = kwargs.pop('nsim', 1000000)
    random_seed = kwargs.pop('random_seed', 0)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # if the implementation errors are uncorrelated with the
    # sampling errrors: var results = var imp + var sampling
    # so std imp = sqrt(var results - var sampling)
    imp_var = (vals_std ** 2) - (bs_std ** 2)
    imp_std = np.sqrt(np.abs(imp_var)) * np.sign(imp_var)
    ind = np.where(imp_std <= 0)[0]
    imp_std[ind] = 0
    imp_std_u = np.zeros(imp_std.shape)
    imp_frac = imp_std / vals_std
    imp_frac_u = np.zeros(imp_frac.shape)
    # Simulate errors distributions
    for i, _ in enumerate(imp_std_u):
        state = np.random.get_state()
        np.random.seed(random_seed)
        sim_vals_std = np.random.normal(vals_std[i], vals_std_u[i], size=nsim)
        sim_bs_std = np.random.normal(bs_std[i], bs_std_u[i], size=nsim)
        sim_imp_var = (sim_vals_std ** 2) - (sim_bs_std ** 2)
        sim_imp_std = np.sqrt(np.abs(sim_imp_var)) * np.sign(sim_imp_var)
        imp_std_u[i] = np.std(sim_imp_std, ddof=1)
        imp_frac_u[i] = np.std((sim_imp_std / sim_vals_std), ddof=1)
        np.random.set_state(state)
    return imp_std, imp_std_u, imp_frac, imp_frac_u