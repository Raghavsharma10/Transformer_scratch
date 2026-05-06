def _M(self, k, t, tips=None, gaps=None):
        """Returns matrix exponential `M`."""
        if self._distributionmodel:
            return self.model.M(k, t, tips, gaps)
        else:
            return self.model.M(t, tips, gaps)