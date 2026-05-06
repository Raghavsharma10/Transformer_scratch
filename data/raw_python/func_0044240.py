def _jdn(self):
        """Return the Julian date number for the given date."""
        if self._last_updated == "gdate":
            return conv.gdate_to_jdn(self.gdate)
        return conv.hdate_to_jdn(self.hdate)