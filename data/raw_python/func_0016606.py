def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k, c in iteritems(self.counts):
            b = max(b, (c, k))
        return b[1]