def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        check_is_fitted(self, 'means_')

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        weight_cdf = np.cumsum(self.weights_)

        X = np.empty((n_samples, self.means_.shape[1]))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                if self.covariance_type == 'tied':
                    cv = self.covars_
                elif self.covariance_type == 'spherical':
                    cv = self.covars_[comp][0]
                else:
                    cv = self.covars_[comp]
                X[comp_in_X] = sample_gaussian(
                    self.means_[comp], cv, self.covariance_type,
                    num_comp_in_X, random_state=random_state).T
        return X