def t(self, value):
        """Set new branch lengths, update likelihood and derivatives."""
        assert (isinstance(value, scipy.ndarray) and (value.dtype ==
                'float') and (value.shape == self.t.shape))
        if (self._t != value).any():
            self._t = value.copy()
            self._updateInternals()