def _compute_biases(self, rs):
        """Generate MLP biases"""

        # use supplied biases if present
        biases = self._get_user_components('biases')
        if (biases is None):
            b_size = self.n_hidden
            biases = rs.normal(size=b_size)

        self.components_['biases'] = biases