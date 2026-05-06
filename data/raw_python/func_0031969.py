def copy(self):
        """Returns a new :class:`~pyinter.Interval` object with the same bounds and values."""
        return Interval(self._lower, self._lower_value, self._upper_value, self._upper)