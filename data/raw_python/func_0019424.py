def fit(self, X):
        """ Fit mixture-density parameters with EM algorithm
        """
        params_dict = _fit_gmm_params(X=X, n_mixtures=self.n_clusters, \
                        n_init=self.n_trials, init_method=self.init_method, \
                        n_iter=self.max_iter, tol=self.tol)
        self.priors_ = params_dict['priors']
        self.means_  = params_dict['means']
        self.covars_ = params_dict['covars']

        self.converged = True
        self.labels_ = self.predict(X)