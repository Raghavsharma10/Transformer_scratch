def dtcurrent(self, value):
        """Set value of `dtcurrent`, update derivatives if needed."""
        assert isinstance(value, bool)
        if value and self.dparamscurrent:
            raise RuntimeError("Can't set both dparamscurrent and dtcurrent True")
        if value != self.dtcurrent:
            self._dtcurrent = value
            self._updateInternals()