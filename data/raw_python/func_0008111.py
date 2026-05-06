def generate_observation_trajectory(self, s_t, dtype=None):
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
        dtype : numpy.dtype, optional, default=None
            The datatype to return the resulting observations in. If None, will select int32.

        Examples
        --------

        Generate an observation model and synthetic state trajectory.

        >>> nobs = 1000
        >>> output_model = DiscreteOutputModel(np.array([[0.5,0.5],[0.1,0.9]]))
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])

        Generate a synthetic trajectory

        >>> o_t = output_model.generate_observation_trajectory(s_t)

        """
        if dtype is None:
            dtype = np.int32

        # Determine number of samples to generate.
        T = s_t.shape[0]
        nsymbols = self._output_probabilities.shape[1]

        if (s_t.max() >= self.nstates) or (s_t.min() < 0):
            msg = ''
            msg += 's_t = %s\n' % s_t
            msg += 's_t.min() = %d, s_t.max() = %d\n' % (s_t.min(), s_t.max())
            msg += 's_t.argmax = %d\n' % s_t.argmax()
            msg += 'self.nstates = %d\n' % self.nstates
            msg += 's_t is out of bounds.\n'
            raise Exception(msg)

        # generate random generators
        # import scipy.stats
        # gens = [scipy.stats.rv_discrete(values=(range(len(self.B[state_index])), self.B[state_index]))
        #         for state_index in range(self.B.shape[0])]
        # o_t = np.zeros([T], dtype=dtype)
        # for t in range(T):
        #     s = s_t[t]
        #     o_t[t] = gens[s].rvs(size=1)
        # return o_t

        o_t = np.zeros([T], dtype=dtype)
        for t in range(T):
            s = s_t[t]
            o_t[t] = np.random.choice(nsymbols, p=self._output_probabilities[s, :])

        return o_t