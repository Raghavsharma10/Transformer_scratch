def root(self, value):
        """Set new XML tree"""

        self._xml = t2s(value)
        self._root = value