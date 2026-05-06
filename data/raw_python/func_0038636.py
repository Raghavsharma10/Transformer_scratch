def search(self, key, default=None):
        """Find the first key-value pair with key *key* and return its value.

        If the key was not found, return *default*. If no default was provided,
        return ``None``. This method never raises a ``KeyError``.
        """
        self._find_lt(key)
        node = self._path[0][2]
        if node is self._tail or key < node[0]:
            return default
        return node[1]