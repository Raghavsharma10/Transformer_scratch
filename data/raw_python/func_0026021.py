def _has(self, key, exact=0):

        """Returns false if key is not found or is ambiguous"""

        if not exact:
            try:
                key = self.getfullkey(key)
                return 1
            except KeyError:
                return 0
        else:
            return key in self.data