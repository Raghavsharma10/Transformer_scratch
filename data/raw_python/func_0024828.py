def evaluate(x, amplitude, mean, stddev):
        """
        GaussianAbsorption1D model function.
        """
        return 1.0 - Gaussian1D.evaluate(x, amplitude, mean, stddev)