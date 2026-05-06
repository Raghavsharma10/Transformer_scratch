def popitem(self):
        """
        Remove and return the item with highest priority. Raises ``KeyError``
        if pqdict is empty.

        """
        heap = self._heap
        position = self._position

        try:
            end = heap.pop(-1)
        except IndexError:
            raise KeyError('pqdict is empty')

        if heap:
            node = heap[0]
            heap[0] = end
            position[end.key] = 0
            self._sink(0)
        else:
            node = end
        del position[node.key]
        return node.key, node.value