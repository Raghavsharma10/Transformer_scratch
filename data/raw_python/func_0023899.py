def enforce_bounds(self, v):
        """Set `enforce_bounds` for both of the kernels to a new value.
        """
        self._enforce_bounds = v
        self.k1.enforce_bounds = v
        self.k2.enforce_bounds = v