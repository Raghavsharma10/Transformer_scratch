def sample(self, observations, prior=None):
        """
        Sample a new set of distribution parameters given a sample of observations from the given state.

        Both the internal parameters and the attached HMM model are updated.

        Parameters
        ----------
        observations :  [ numpy.array with shape (N_k,) ] with `nstates` elements
            observations[k] is a set of observations sampled from state `k`
        prior : object
            prior option for compatibility

        Examples
        --------

        Generate synthetic observations.

        >>> nstates = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=nstates, means=[-1, 0, 1], sigmas=[0.5, 1, 2])
        >>> observations = [ output_model.generate_observations_from_state(state_index, nobs) for state_index in range(nstates) ]

        Update output parameters by sampling.

        >>> output_model.sample(observations)

        """
        for state_index in range(self.nstates):
            # Update state emission distribution parameters.

            observations_in_state = observations[state_index]
            # Determine number of samples in this state.
            nsamples_in_state = len(observations_in_state)

            # Skip update if no observations.
            if nsamples_in_state == 0:
                logger().warn('Warning: State %d has no observations.' % state_index)
            if nsamples_in_state > 0:  # Sample new mu.
                self.means[state_index] = np.random.randn()*self.sigmas[state_index]/np.sqrt(nsamples_in_state) + np.mean(observations_in_state)
            if nsamples_in_state > 1:  # Sample new sigma
                # This scheme uses the improper Jeffreys prior on sigma^2, P(mu, sigma^2) \propto 1/sigma
                chisquared = np.random.chisquare(nsamples_in_state-1)
                sigmahat2 = np.mean((observations_in_state - self.means[state_index])**2)
                self.sigmas[state_index] = np.sqrt(sigmahat2) / np.sqrt(chisquared / nsamples_in_state)

        return