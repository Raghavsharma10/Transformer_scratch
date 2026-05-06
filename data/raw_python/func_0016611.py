def probability_gt(self, x):
        """
        Returns the probability of a random variable being greater than the
        given value.
        """
        if self.mean is None:
            return
        p = normdist(x=x, mu=self.mean, sigma=self.standard_deviation)
        return 1-p