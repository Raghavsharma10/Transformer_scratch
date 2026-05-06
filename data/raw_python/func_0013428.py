def _removeLru(self):
        """
        Remove the least recently used file handle from the cache.
        The pop method removes an element from the right of the deque.
        Returns the name of the file that has been removed.
        """
        (dataFile, handle) = self._cache.pop()
        handle.close()
        return dataFile