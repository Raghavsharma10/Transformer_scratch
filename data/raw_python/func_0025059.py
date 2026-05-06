def compute_weight(self, r, ytr=None, mask=None):
        """Returns the weight (w) using OLS of r * w = gp._ytr """
        ytr = self._ytr if ytr is None else ytr
        mask = self._mask if mask is None else mask
        return compute_weight(r, ytr, mask)