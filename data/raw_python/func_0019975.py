def stats(self):
        """
        The first four standard moments of a distribution: mean, variance, and
        standardized skewness and kurtosis coefficients.
        """
        mn = self.mean
        vr = self.var
        sk = self.skew
        kt = self.kurt
        return [mn, vr, sk, kt]