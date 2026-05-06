def getallkeys(self, key, failobj=None):
        """Returns a list of the full key names (not the items)
        for all the matching values for key.  The list will
        contain a single entry for unambiguous matches and
        multiple entries for ambiguous matches."""
        if self.mmkeys is None: self._mmInit()
        return self.mmkeys.get(key, failobj)