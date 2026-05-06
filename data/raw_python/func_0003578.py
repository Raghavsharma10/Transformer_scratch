def get_row(self, index, default=None):
        """Return the row at the given index or the default value."""
        if not isinstance(index, int) or index < 0 or index >= len(self._rows):
            return default
        return self._rows[index]