def fit(self, X, y=None):
        """
        Sets *self.degree* according to :func:`baart_criteria` if *degree_range*
        is not None, otherwise does nothing.

        **Parameters**

        X : array-like, shape = [n_samples, 1]
            Column vector of phases.
        y : array-like or None, shape = [n_samples], optional
            Row vector of magnitudes (default None).

        **Returns**

        self : returns an instance of self
        """
        if self.degree_range is not None:
            self.degree = self.baart_criteria(X, y)
        return self