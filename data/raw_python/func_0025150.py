def topitem(self):
        """
        Return the item with highest priority. Raises ``KeyError`` if pqdict is
        empty.

        """
        try:
            node = self._heap[0]
        except IndexError:
            raise KeyError('pqdict is empty')
        return node.key, node.value