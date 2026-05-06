def gdate(self):
        """Return the Gregorian date for the given Hebrew date object."""
        if self._last_updated == "gdate":
            return self._gdate
        return conv.jdn_to_gdate(self._jdn)