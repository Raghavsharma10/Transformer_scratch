def drop(self, names):
        """Drops variables (names) from metadata."""
        
        # drop lower dimension data
        self._data = self._data.drop(names, axis=0)
        # drop higher dimension data
        for name in names:
            if name in self._ho_data:
                _ = self._ho_data.pop(name)