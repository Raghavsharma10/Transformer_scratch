def index(self, key, default=UNSET):
        """Find the first key-value pair with key *key* and return its position.

        If the key is not found, return *default*. If default was not provided,
        raise a ``KeyError``
        """
        self._find_lt(key)
        node = self._path[0][2]
        if node is self._tail or key < node[0]:
            if default is self.UNSET:
                raise KeyError('key {!r} not in list'.format(key))
            return default
        return self._distance[0]