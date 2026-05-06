def _update_model(self, gammas, count_matrices, maxiter=10000000):
        """
        Maximization step: Updates the HMM model given the hidden state assignment and count matrices

        Parameters
        ----------
        gamma : [ ndarray(T,N, dtype=float) ]
            list of state probabilities for each trajectory
        count_matrix : [ ndarray(N,N, dtype=float) ]
            list of the Baum-Welch transition count matrices for each hidden
            state trajectory
        maxiter : int
            maximum number of iterations of the transition matrix estimation if
            an iterative method is used.

        """
        gamma0_sum = self._init_counts(gammas)
        C = self._transition_counts(count_matrices)
        logger().info("Initial count = \n"+str(gamma0_sum))
        logger().info("Count matrix = \n"+str(C))

        # compute new transition matrix
        from bhmm.estimators._tmatrix_disconnected import estimate_P, stationary_distribution
        T = estimate_P(C, reversible=self._hmm.is_reversible, fixed_statdist=self._fixed_stationary_distribution,
                       maxiter=maxiter, maxerr=1e-12, mincount_connectivity=1e-16)
        # print 'P:\n', T
        # estimate stationary or init distribution
        if self._stationary:
            if self._fixed_stationary_distribution is None:
                pi = stationary_distribution(T, C=C, mincount_connectivity=1e-16)
            else:
                pi = self._fixed_stationary_distribution
        else:
            if self._fixed_initial_distribution is None:
                pi = gamma0_sum / np.sum(gamma0_sum)
            else:
                pi = self._fixed_initial_distribution
        # print 'pi: ', pi, ' stationary = ', self._hmm.is_stationary

        # update model
        self._hmm.update(pi, T)

        logger().info("T: \n"+str(T))
        logger().info("pi: \n"+str(pi))

        # update output model
        self._hmm.output_model.estimate(self._observations, gammas)