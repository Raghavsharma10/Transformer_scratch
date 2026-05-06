def popitem(self):
        """Removes the first key-value pair and return it.

        This method raises a ``KeyError`` if the list is empty.
        """
        node = self._head[2]
        if node is self._tail:
            raise KeyError('list is empty')
        self._find_lt(node[0])
        self._remove(node)
        return (node[0], node[1])