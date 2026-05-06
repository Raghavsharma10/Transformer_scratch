def lines(self):
        """List of file lines."""
        if self._lines is None:
            with io.open(self.path, 'r', encoding='utf-8') as fh:
                self._lines = fh.read().split('\n')

        return self._lines