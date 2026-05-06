def fit(self, X, y=None):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        # initialization step
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        for _ in range(self.n_init):
            if 'm' in self.init_params or not hasattr(self, 'means_'):
                if np.issubdtype(X.dtype, np.float32):
                    from bhmm._external.clustering.kmeans_clustering_32 import init_centers
                elif np.issubdtype(X.dtype, np.float64):
                    from bhmm._external.clustering.kmeans_clustering_64 import init_centers
                else:
                    raise ValueError("Could not handle dtype %s for clustering!" % X.dtype)
                centers = init_centers(X, 'euclidean', self.n_components)
                self.means_ = centers

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)

            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = \
                    distribute_covar_matrix_to_match_covariance_type(
                        cv, self.covariance_type, self.n_components)

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            # this line should be removed when 'thresh' is removed in v0.18
            tol = (self.tol if self.thresh is None
                   else self.thresh / float(X.shape[0]))

            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self.score_samples(X)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                # (should compare to self.tol when dreprecated 'thresh' is
                # removed in v0.18)
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if change < tol:
                        self.converged_ = True
                        break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")
        # self.n_iter == 0 occurs when using GMM within HMM
        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        return self