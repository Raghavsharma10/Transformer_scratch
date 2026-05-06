def temperature(self, what):
        """Set temperature."""
        self._temperature = units.validate_quantity(what, u.K)