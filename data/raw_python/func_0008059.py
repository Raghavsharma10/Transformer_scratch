def _do_mstep(self, X, responsibilities, params, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weihgts.
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        if 'c' in params:
            covar_mstep_func = _covar_mstep_funcs[self.covariance_type]
            self.covars_ = covar_mstep_func(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        return weights