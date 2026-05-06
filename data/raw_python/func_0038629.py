def insert(self, key, value):
        """Insert a key-value pair in the list.

        The pair is inserted at the correct location so that the list remains
        sorted on *key*. If a pair with the same key is already in the list,
        then the pair is appended after all other pairs with that key.
        """
        self._find_lte(key)
        node = self._create_node(key, value)
        self._insert(node)