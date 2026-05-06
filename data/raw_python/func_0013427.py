def _update(self, dataFile, handle):
        """
        Update the priority of the file handle. The element is first
        removed and then added to the left of the deque.
        """
        self._cache.remove((dataFile, handle))
        self._add(dataFile, handle)