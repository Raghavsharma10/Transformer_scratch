def updateitem(self, key, new_val):
        """
        Update the priority value of an existing item. Raises ``KeyError`` if
        key is not in the pqdict.

        """
        if key not in self._position:
            raise KeyError(key)
        self[key] = new_val