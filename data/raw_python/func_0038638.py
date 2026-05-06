def pop(self, key, default=UNSET):
        """Remove the first key-value pair with key *key*.

        If a pair was removed, return its value. Otherwise if *default* was
        provided, return *default*. Otherwise a ``KeyError`` is raised.
        """
        self._find_lt(key)
        node = self._path[0][2]
        if node is self._tail or key < node[0]:
            if default is self.UNSET:
                raise KeyError('key {!r} not in list')
            return default
        self._remove(node)
        return node[1]