def swap_priority(self, key1, key2):
        """
        Fast way to swap the priority level of two items in the pqdict. Raises
        ``KeyError`` if either key does not exist.

        """
        heap = self._heap
        position = self._position
        if key1 not in self or key2 not in self:
            raise KeyError
        pos1, pos2 = position[key1], position[key2]
        heap[pos1].key, heap[pos2].key = key2, key1
        position[key1], position[key2] = pos2, pos1