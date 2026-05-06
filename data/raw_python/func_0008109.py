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

        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))

        Generate sample from each state.

        >>> observation = output_model.generate_observation_from_state(0)

        """
        # generate random generator (note that this is inefficient - better use one of the next functions
        import scipy.stats
        gen = scipy.stats.rv_discrete(values=(range(len(self._output_probabilities[state_index])), 
                                              self._output_probabilities[state_index]))
        gen.rvs(size=1)