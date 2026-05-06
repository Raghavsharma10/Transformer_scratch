def iteritems(self, pipe=None):
        """Return an iterator over the dictionary's ``(key, value)`` pairs."""
        pipe = self.redis if pipe is None else pipe
        for k, v in self._data(pipe).items():
            yield k, self.cache.get(k, v)