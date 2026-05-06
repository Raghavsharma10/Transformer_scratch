def mu(self, value):
        """Set new `mu` value."""
        _checkParam('mu', value, self.PARAMLIMITS, self.PARAMTYPES)
        if value != self.mu:
            self._cached = {}
        self._mu = value