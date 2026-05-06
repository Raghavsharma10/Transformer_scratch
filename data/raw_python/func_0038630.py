def replace(self, key, value):
        """Replace the value of the first key-value pair with key *key*.

        If the key was not found, the pair is inserted.
        """
        self._find_lt(key)
        node = self._path[0][2]
        if node is self._tail or key < node[0]:
            node = self._create_node(key, value)
            self._insert(node)
        else:
            node[1] = value