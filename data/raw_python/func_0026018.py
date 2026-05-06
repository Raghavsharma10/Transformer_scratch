def getall(self, key, failobj=None):
        """Returns a list of all the matching values for key,
        containing a single entry for unambiguous matches and
        multiple entries for ambiguous matches."""
        if self.mmkeys is None: self._mmInit()
        k = self.mmkeys.get(key)
        if not k: return failobj
        return list(map(self.data.get, k))