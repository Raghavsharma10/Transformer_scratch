def fit_deriv(x, amplitude, mean, stddev):
        """
        GaussianAbsorption1D model function derivatives.
        """
        import operator
        return list(map(
            operator.neg, Gaussian1D.fit_deriv(x, amplitude, mean, stddev)))