def check(self):
        """Check that the entry has the required fields."""
        # Make sure there is a schema key in dict
        if self._KEYS.SCHEMA not in self:
            self[self._KEYS.SCHEMA] = self.catalog.SCHEMA.URL
        # Make sure there is a name key in dict
        if (self._KEYS.NAME not in self or len(self[self._KEYS.NAME]) == 0):
            raise ValueError("Entry name is empty:\n\t{}".format(
                json.dumps(
                    self, indent=2)))
        return