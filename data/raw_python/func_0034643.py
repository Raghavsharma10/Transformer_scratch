def xml(self, value):
        """Set new XML string"""

        self._xml = value
        self._root = s2t(value)