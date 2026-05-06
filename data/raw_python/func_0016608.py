def update(self, dist):
        """
        Adds the given distribution's counts to the current distribution.
        """
        assert isinstance(dist, DDist)
        for k, c in iteritems(dist.counts):
            self.counts[k] += c
        self.total += dist.total