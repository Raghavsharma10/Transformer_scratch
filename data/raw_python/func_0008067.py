def estimate(self, observations, weights):
        """
        Fits the output model given the observations and weights

        Parameters
        ----------
        observations : [ ndarray(T_k,) ] with K elements
            A list of K observation trajectories, each having length T_k and d dimensions
        weights : [ ndarray(T_k,nstates) ] with K elements
            A list of K weight matrices, each having length T_k
            weights[k][t,n] is the weight assignment from observations[k][t] to state index n

        Examples
        --------

        Generate an observation model and samples from each state.

        >>> ntrajectories = 3
        >>> nobs = 1000
        >>> output_model = GaussianOutputModel(nstates=3, means=[-1, 0, +1], sigmas=[0.5, 1, 2])
        >>> observations = [ np.random.randn(nobs) for _ in range(ntrajectories) ] # random observations
        >>> weights = [ np.random.dirichlet([2, 3, 4], size=nobs) for _ in range(ntrajectories) ] # random weights

        Update the observation model parameters my a maximum-likelihood fit.

        >>> output_model.estimate(observations, weights)

        """
        # sizes
        N = self.nstates
        K = len(observations)

        # fit means
        self._means = np.zeros(N)
        w_sum = np.zeros(N)
        for k in range(K):
            # update nominator
            for i in range(N):
                self.means[i] += np.dot(weights[k][:, i], observations[k])
            # update denominator
            w_sum += np.sum(weights[k], axis=0)
        # normalize
        self._means /= w_sum

        # fit variances
        self._sigmas = np.zeros(N)
        w_sum = np.zeros(N)
        for k in range(K):
            # update nominator
            for i in range(N):
                Y = (observations[k] - self.means[i])**2
                self.sigmas[i] += np.dot(weights[k][:, i], Y)
            # update denominator
            w_sum += np.sum(weights[k], axis=0)
        # normalize
        self._sigmas /= w_sum
        self._sigmas = np.sqrt(self.sigmas)
        if np.any(self._sigmas < np.finfo(self._sigmas.dtype).eps):
            raise RuntimeError('at least one sigma is too small to continue.')