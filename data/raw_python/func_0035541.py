def stationarystate(self, k):
        """See docs for `Model` abstract base class."""
        assert 0 <= k < self.ncats
        return self._models[k].stationarystate