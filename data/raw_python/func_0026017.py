def _has(self, key, exact=0):
        """Raises an exception if key is ambiguous"""
        if not exact:
            key = self.getfullkey(key,new=1)
        return key in self.data