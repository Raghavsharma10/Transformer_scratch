def generate_observations_from_state(self, state_index, nobs):
        """
        Generate synthetic observation data from a given state.

        Parameters
        ----------
        state_index : int
            Index of the state from which observations are to be generated.
        nobs : int
            The number of observations to generate.

        Returns
        -------
        observations : numpy.array of shape(nobs,) with type dtype
            A sample of `nobs` observations from the specified state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))

        Generate sample from each state.

        >>> observations = [output_model.generate_observations_from_state(state_index, nobs=100) for state_index in range(output_model.nstates)]

        """
        import scipy.stats
        gen = scipy.stats.rv_discrete(values=(range(self._nsymbols), self._output_probabilities[state_index]))
        gen.rvs(size=nobs)