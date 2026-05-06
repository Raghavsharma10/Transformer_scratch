def perturbed_trajectory(trajectory, sensitivity_trajectory, delta=1e-4):
    """
    Slightly perturb trajectory wrt the parameter specified in sensitivity_trajectory.

    :param trajectory: the actual trajectory for an ODE term
    :type trajectory: :class:`Trajectory`
    :param sensitivity_trajectory: sensitivity trajectory (dy/dpi for all timepoints t)
    :type sensitivity_trajectory: :class:`Trajectory`
    :param delta: the perturbation size
    :type delta: float
    :return: :class:`Trajectory`
    """
    sensitivity_trajectory_description = sensitivity_trajectory.description
    assert(isinstance(sensitivity_trajectory_description, SensitivityTerm))
    assert(np.equal(trajectory.timepoints, sensitivity_trajectory.timepoints).all())

    return Trajectory(trajectory.timepoints,
                      trajectory.values + sensitivity_trajectory.values * delta,
                      PerturbedTerm(sensitivity_trajectory_description.ode_term,
                                    sensitivity_trajectory_description.parameter,
                                    delta))