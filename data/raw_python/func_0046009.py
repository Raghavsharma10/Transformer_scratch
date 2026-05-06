def generate_single_simulation(self, x):
        """
        Generate a single SSA simulation
        :param x: an integer to reset the random seed. If None, the initial random number generator is used
        :return: a list of :class:`~means.simulation.Trajectory` one per species in the problem
        :rtype: list[:class:`~means.simulation.Trajectory`]
        """
        #reset random seed
        if x:
            self.__rng = np.random.RandomState(x)

        # perform one stochastic simulation
        time_points, species_over_time = self._gssa(self.__initial_conditions, self.__t_max)

        # build descriptors for first order raw moments aka expectations (e.g. [1, 0, 0], [0, 1, 0] and [0, 0, 1])
        descriptors = []
        for i, s in enumerate(self.__species):
            row = [0] * len(self.__species)
            row[i] = 1
            descriptors.append(Moment(row, s))

        # build trajectories
        trajectories = [Trajectory(time_points, spot, desc) for
                        spot, desc in zip(species_over_time, descriptors)]

        return trajectories