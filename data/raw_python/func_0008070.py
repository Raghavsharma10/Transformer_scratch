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
        observations : numpy.array of shape(nobs,)
            A sample of `nobs` observations from the specified state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = GaussianOutputModel(nstates=2, means=[0, 1], sigmas=[1, 2])

        Generate samples from each state.

        >>> observations = [ output_model.generate_observations_from_state(state_index, nobs=100) for state_index in range(output_model.nstates) ]

        """
        observations = self.sigmas[state_index] * np.random.randn(nobs) + self.means[state_index]
        return observations