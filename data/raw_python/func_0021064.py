def _data(self, pipe=None):
        """
        Return a :obj:`list` of all values from Redis
        (without checking the local cache).
        """
        pipe = self.redis if pipe is None else pipe
        return [self._unpickle(v) for v in pipe.lrange(self.key, 0, -1)]