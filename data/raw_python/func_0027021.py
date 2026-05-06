def get(self, key, default=miss):
        """Return the value for given key if it exists."""
        if key not in self._dict:
            return default

        # invokes __getitem__, which updates the item
        return self[key]