def _set_covars(self, covars):
        """Provide values for covariance"""
        covars = np.asarray(covars)
        _validate_covars(covars, self.covariance_type, self.n_components)
        self.covars_ = covars