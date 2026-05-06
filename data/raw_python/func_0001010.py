def _generate_components(self, X):
        """Generate components of hidden layer given X"""

        rs = check_random_state(self.random_state)
        if (self._use_mlp_input):
            self._compute_biases(rs)
            self._compute_weights(X, rs)

        if (self._use_rbf_input):
            self._compute_centers(X, sp.issparse(X), rs)
            self._compute_radii()