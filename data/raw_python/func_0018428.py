def cursor(self):
        """The position of the cursor in the text."""
        if self._cursor < 0:
            self.cursor = 0

        if self._cursor > len(self):
            self.cursor = len(self)

        return self._cursor