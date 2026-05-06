def generate_observation_from_state(self, state_index):
        """
        Generate a single synthetic observation data from a given state.

        Parameters
        ----------
        state_index : int
            Index of the state from which observations are to be generated.

        Returns
        -------
        observation : float
            A single observation from the given state.

        Examples
        --------

        Generate an observation model.

        >>> output_model = GaussianOutputModel(nstates=2, means=[0, 1], sigmas=[1, 2])

        Generate sample from a state.

        >>> observation = output_model.generate_observation_from_state(0)

        """
        observation = self.sigmas[state_index] * np.random.randn() + self.means[state_index]
        return observation