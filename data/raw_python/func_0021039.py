def remove(self, value):
        """
        Remove element *value* from the set. Raises :exc:`KeyError` if it
        is not contained in the set.
        """
        # Raise TypeError if value is not hashable
        hash(value)

        result = self.redis.srem(self.key, self._pickle(value))
        if not result:
            raise KeyError(value)