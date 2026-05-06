def replace_key(self, key, new_key):
        """
        Replace the key of an existing heap node in place. Raises ``KeyError``
        if the key to replace does not exist or if the new key is already in
        the pqdict.

        """
        heap = self._heap
        position = self._position
        if new_key in self:
            raise KeyError('%s is already in the queue' % repr(new_key))
        pos = position.pop(key)  # raises appropriate KeyError
        position[new_key] = pos
        heap[pos].key = new_key