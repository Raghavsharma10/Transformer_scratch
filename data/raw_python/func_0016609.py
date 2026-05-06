def probability_lt(self, x):
        """
        Returns the probability of a random variable being less than the
        given value.
        """
        if self.mean is None:
            return
        return normdist(x=x, mu=self.mean, sigma=self.standard_deviation)