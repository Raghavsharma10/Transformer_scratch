def remove(self, key):
        """Remove the first key-value pair with key *key*.

        If the key was not found, a ``KeyError`` is raised.
        """
        self._find_lt(key)
        node = self._path[0][2]
        if node is self._tail or key < node[0]:
            raise KeyError('{!r} is not in list'.format(key))
        self._remove(node)