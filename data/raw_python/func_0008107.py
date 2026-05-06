def estimate(self, observations, weights):
        """
        Maximum likelihood estimation of output model given the observations and weights

        Parameters
        ----------

        observations : [ ndarray(T_k) ] with K elements
            A list of K observation trajectories, each having length T_k
        weights : [ ndarray(T_k, N) ] with K elements
            A list of K weight matrices, each having length T_k and containing the probability of any of the states in
            the given time step

        Examples
        --------

        Generate an observation model and samples from each state.

        >>> import numpy as np
        >>> ntrajectories = 3
        >>> nobs = 1000
        >>> B = np.array([[0.5,0.5],[0.1,0.9]])
        >>> output_model = DiscreteOutputModel(B)

        >>> from scipy import stats
        >>> nobs = 1000
        >>> obs = np.empty(nobs, dtype = object)
        >>> weights = np.empty(nobs, dtype = object)

        >>> gens = [stats.rv_discrete(values=(range(len(B[i])), B[i])) for i in range(B.shape[0])]
        >>> obs = [gens[i].rvs(size=nobs) for i in range(B.shape[0])]
        >>> weights = [np.zeros((nobs, B.shape[1])) for i in range(B.shape[0])]
        >>> for i in range(B.shape[0]): weights[i][:, i] = 1.0

        Update the observation model parameters my a maximum-likelihood fit.

        >>> output_model.estimate(obs, weights)

        """
        # sizes
        N, M = self._output_probabilities.shape
        K = len(observations)
        # initialize output probability matrix
        self._output_probabilities = np.zeros((N, M))
        # update output probability matrix (numerator)
        if self.__impl__ == self.__IMPL_C__:
            for k in range(K):
                dc.update_pout(observations[k], weights[k], self._output_probabilities, dtype=config.dtype)
        elif self.__impl__ == self.__IMPL_PYTHON__:
            for k in range(K):
                for o in range(M):
                    times = np.where(observations[k] == o)[0]
                    self._output_probabilities[:, o] += np.sum(weights[k][times, :], axis=0)
        else:
            raise RuntimeError('Implementation '+str(self.__impl__)+' not available')
        # normalize
        self._output_probabilities /= np.sum(self._output_probabilities, axis=1)[:, None]