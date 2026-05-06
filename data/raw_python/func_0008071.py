def generate_observation_trajectory(self, s_t):
        """
        Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        s_t : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]

        Examples
        --------

        Generate an observation model and synthetic state trajectory.

        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """

        # Determine number of samples to generate.
        T = s_t.shape[0]

        o_t = np.zeros([T], dtype=config.dtype)
        for t in range(T):
            s = s_t[t]
            o_t[t] = self.sigmas[s] * np.random.randn() + self.means[s]
        return o_t