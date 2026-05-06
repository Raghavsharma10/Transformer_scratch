def text(self):
        """Return the string to render."""
        if callable(self._text):
            return str(self._text())
        return str(self._text)