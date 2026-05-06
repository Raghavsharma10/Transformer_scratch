def _data(self, pipe=None):
        """
        Returns a Python dictionary with the same values as this object
        (without checking the local cache).
        """
        pipe = self.redis if pipe is None else pipe
        items = pipe.hgetall(self.key).items()

        return {self._unpickle_key(k): self._unpickle(v) for k, v in items}