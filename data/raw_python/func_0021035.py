def discard(self, value):
        """Remove element *value* from the set if it is present."""
        # Raise TypeError if value is not hashable
        hash(value)

        self.redis.srem(self.key, self._pickle(value))