def M(self, k, t, tips=None, gaps=None):
        """See docs for `DistributionModel` abstract base class."""
        assert 0 <= k < self.ncats
        return self._models[k].M(t, tips=tips, gaps=gaps)