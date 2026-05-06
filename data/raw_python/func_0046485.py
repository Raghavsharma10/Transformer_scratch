def _distribution_distance(simulated_trajectories, observed_trajectories_lookup, distribution):
    """
    Returns the distance between the simulated and observed trajectory, w.r.t. the assumed distribution

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :param distribution: Distribution to use. See :func:`_eval_density` for the list of available distributions
    :return:
    """

    mean_variance_lookup = _compile_mean_variance_lookup(simulated_trajectories)

    # get moment expansion result with current parameters
    log_likelihood = 0

    for trajectory in observed_trajectories_lookup.itervalues():
        moment = trajectory.description
        assert(isinstance(moment, Moment))
        assert(moment.order == 1)

        species = np.where(moment.n_vector == 1)[0][0]
        mean_variance = mean_variance_lookup[species]
        if (mean_variance.mean < 0).any() or (mean_variance.variance < 0).any():
            return float('inf')

        term = _eval_density(mean_variance.mean, mean_variance.variance, trajectory.values, distribution)
        log_likelihood += term

    dist = -log_likelihood
    return dist