def simulate_system(self, parameters, initial_conditions, timepoints,
                        max_moment_order=1, number_of_processes=1):
        """
        Perform Gillespie SSA simulations and returns trajectories for of each species.
        Each trajectory is interpolated at the given time points.
        By default, the average amounts of species for all simulations is returned.

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem.
                        Must be in the same order as these equations occur.

        :param timepoints: A list of time points to simulate the system for

        :param number_of_processes: the number of parallel process to be run
        :param max_moment_order: the highest moment order to calculate the trajectories to.
                                 if set to zero, the individual trajectories will be returned, instead of
                                 the averaged moments.
        E.g. a value of one will return means, a values of two, means, variances and covariance and so on.


        :return: a list of :class:`~means.simulation.Trajectory` one per species in the problem,
            or a list of lists of trajectories (one per simulation) if `return_average == False`.
        :rtype: list[:class:`~means.simulation.Trajectory`]
        """
        max_moment_order = int(max_moment_order)
        assert(max_moment_order >= 0)

        n_simulations = self.__n_simulations
        self._validate_parameters(parameters, initial_conditions)
        t_max= max(timepoints)

        substitution_pairs = dict(zip(self.__problem.parameters, parameters))
        propensities = substitute_all(self.__problem.propensities, substitution_pairs)
        # lambdify the propensities for fast evaluation
        propensities_as_function = self.__problem.propensities_as_function
        def f(*species_parameters):
            return propensities_as_function(*(np.concatenate((species_parameters, parameters))))

        population_rates_as_function = f

        if not self.__random_seed:
            seed_for_processes = [None] * n_simulations
        else:
            seed_for_processes = [i for i in range(self.__random_seed, n_simulations + self.__random_seed)]



        if number_of_processes ==1:
            ssa_generator = _SSAGenerator(population_rates_as_function,
                                        self.__problem.change, self.__problem.species,
                                        initial_conditions, t_max, seed=self.__random_seed)

            results = map(ssa_generator.generate_single_simulation, seed_for_processes)


        else:
            p = multiprocessing.Pool(number_of_processes,
                    initializer=multiprocessing_pool_initialiser,
                    initargs=[population_rates_as_function, self.__problem.change,
                              self.__problem.species,
                              initial_conditions, t_max, self.__random_seed])

            results = p.map(multiprocessing_apply_ssa, seed_for_processes)

            p.close()
            p.join()

        resampled_results = [[traj.resample(timepoints, extrapolate=True) for traj in res] for res in results]
        for i in resampled_results:
            idx = len(i[0].values) - 1

        if max_moment_order == 0:
            # Return a list of TrajectoryCollection objects
            return map(TrajectoryCollection, resampled_results)

        moments = self._compute_moments(resampled_results, max_moment_order)
        return TrajectoryCollection(moments)