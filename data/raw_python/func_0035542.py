def dstationarystate(self, k, param):
        """See docs for `Model` abstract base class."""
        assert param not in self.distributionparams
        assert param in self.freeparams or param == self.distributedparam
        ds = self._models[k].dstationarystate(param)
        return ds