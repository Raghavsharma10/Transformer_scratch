def clear(self):
        """Clear the console"""
        if hasattr(self, '_bytes_012'):
            self._bytes_012.fill(0)
            self._bytes_345.fill(0)
        self._text_lines = [] * self._n_rows
        self._pending_writes = []