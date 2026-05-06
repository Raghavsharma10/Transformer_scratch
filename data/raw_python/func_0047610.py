def simulate_system(self, parameters, initial_conditions, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem. Must be in the same order as
                               these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return: a list of :class:`~means.simulation.TrajectoryWithSensitivityData` objects,
                 one for each of the equations in the problem
        :rtype: list[:class:`~means.simulation.TrajectoryWithSensitivityData`]
        """
        return super(SimulationWithSensitivities, self).simulate_system(parameters, initial_conditions, timepoints)