def sum_of_squares(simulated_trajectories, observed_trajectories_lookup):
    """
    Returns the sum-of-squares distance between the simulated_trajectories and observed_trajectories

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :return: the distance between simulated and observed trajectories
    :rtype: float
    """
    dist = 0
    for simulated_trajectory in simulated_trajectories:
        observed_trajectory = None
        try:
            observed_trajectory = observed_trajectories_lookup[simulated_trajectory.description]
        except KeyError:
            continue

        deviations = observed_trajectory.values - simulated_trajectory.values
        # Drop NaNs arising from missing datapoints
        deviations = deviations[~np.isnan(deviations)]

        dist += np.sum(np.square(deviations))

    return dist