def _get_covars(self):
        """Covariance parameters for each mixture component.
        The shape depends on `cvtype`::

            (`n_states`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)              if 'tied',
            (`n_states`, `n_features`)                if 'diag',
            (`n_states`, `n_features`, `n_features`)  if 'full'
            """
        if self.covariance_type == 'full':
            return self.covars_
        elif self.covariance_type == 'diag':
            return [np.diag(cov) for cov in self.covars_]
        elif self.covariance_type == 'tied':
            return [self.covars_] * self.n_components
        elif self.covariance_type == 'spherical':
            return [np.diag(cov) for cov in self.covars_]