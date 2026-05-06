def _p_o(self, o):
        """
        Returns the output probability for symbol o from all hidden states

        Parameters
        ----------
        o : float
            A single observation.

        Return
        ------
        p_o : ndarray (N)
            p_o[i] is the probability density of the observation o from state i emission distribution

        Examples
        --------

        Create an observation model.

        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, 1], sigmas=[0.5, 1, 2])

        Compute the output probability of a single observation from all hidden states.

        >>> observation = 0
        >>> p_o = output_model._p_o(observation)

        """
        if self.__impl__ == self.__IMPL_C__:
            return gc.p_o(o, self.means, self.sigmas, out=None, dtype=type(o))
        elif self.__impl__ == self.__IMPL_PYTHON__:
            if np.any(self.sigmas < np.finfo(self.sigmas.dtype).eps):
                raise RuntimeError('at least one sigma is too small to continue.')
            C = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigmas)
            Pobs = C * np.exp(-0.5 * ((o-self.means)/self.sigmas)**2)
            return Pobs
        else:
            raise RuntimeError('Implementation '+str(self.__impl__)+' not available')