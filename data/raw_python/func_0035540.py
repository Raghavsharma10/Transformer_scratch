def mu(self, value):
        """Set new `mu` value."""
        for k in range(self.ncats):
            self._models[k].updateParams({'mu':value})