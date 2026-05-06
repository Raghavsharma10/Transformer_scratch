def sample(self, observations_by_state):
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        The internal parameters are updated.

        Parameters
        ----------
        observations :  [ numpy.array with shape (N_k,) ] with nstates elements
            observations[k] are all observations associated with hidden state k

        Examples
        --------

        initialize output model

        >>> B = np.array([[0.5, 0.5], [0.1, 0.9]])
        >>> output_model = DiscreteOutputModel(B)

        sample given observation

        >>> obs = [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        >>> output_model.sample(obs)

        """
        from numpy.random import dirichlet
        N, M = self._output_probabilities.shape  # nstates, nsymbols
        for i, obs_by_state in enumerate(observations_by_state):
            # count symbols found in data
            count = np.bincount(obs_by_state, minlength=M).astype(float)
            # sample dirichlet distribution
            count += self.prior[i]
            positive = count > 0
            # if counts at all: can't sample, so leave output probabilities as they are.
            self._output_probabilities[i, positive] = dirichlet(count[positive])