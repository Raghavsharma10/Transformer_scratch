def probability_in(self, a, b):
        """
        Returns the probability of a random variable falling between the given
        values.
        """
        if self.mean is None:
            return
        p1 = normdist(x=a, mu=self.mean, sigma=self.standard_deviation)
        p2 = normdist(x=b, mu=self.mean, sigma=self.standard_deviation)
        return abs(p1 - p2)