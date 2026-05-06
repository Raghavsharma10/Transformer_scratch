def dM(self, k, t, param, Mkt, tips=None, gaps=None):
        """See docs for `DistributionModel` abstract base class."""
        assert 0 <= k < self.ncats
        assert ((param in self.freeparams) or (param == 't') or (
                param == self.distributedparam))
        assert param not in self.distributionparams
        return self._models[k].dM(t, param, Mkt, tips=tips, gaps=gaps)