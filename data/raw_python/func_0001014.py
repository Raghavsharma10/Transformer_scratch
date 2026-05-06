def _fit_regression(self, y):
        """
        fit regression using pseudo-inverse
        or supplied regressor
        """

        if self.regressor is None:
            self.coefs_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)
        else:
            self.regressor.fit(self.hidden_activations_, y)

        self.fitted_ = True