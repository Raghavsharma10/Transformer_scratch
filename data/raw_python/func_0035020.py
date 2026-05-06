def _dM(self, k, t, param, M, tips=None, gaps=None):
        """Returns derivative of matrix exponential."""
        if self._distributionmodel:
            return self.model.dM(k, t, param, M, tips, gaps)
        else:
            return self.model.dM(t, param, M, tips, gaps)