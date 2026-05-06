def p_obs(self, obs, out=None):
        """
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        oobs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states

        Examples
        --------

        Generate an observation model and synthetic observation trajectory.

        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> s_t = np.random.randint(0, output_model.nstates, size=[nobs])
        >>> o_t = output_model.generate_observation_trajectory(s_t)

        Compute output probabilities for entire trajectory and all hidden states.

        >>> p_o = output_model.p_obs(o_t)

        """
        if self.__impl__ == self.__IMPL_C__:
            res = gc.p_obs(obs, self.means, self.sigmas, out=out, dtype=config.dtype)
            return self._handle_outliers(res)
        elif self.__impl__ == self.__IMPL_PYTHON__:
            T = len(obs)
            if out is None:
                res = np.zeros((T, self.nstates), dtype=config.dtype)
            else:
                res = out
            for t in range(T):
                res[t, :] = self._p_o(obs[t])
            return self._handle_outliers(res)
        else:
            raise RuntimeError('Implementation '+str(self.__impl__)+' not available')