def pop(self):
        """
        Remove and return an arbitrary element from the set.
        Raises :exc:`KeyError` if the set is empty.
        """
        result = self.redis.spop(self.key)
        if result is None:
            raise KeyError

        return self._unpickle(result)