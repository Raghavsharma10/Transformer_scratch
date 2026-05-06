def add(self, value):
        """Add element *value* to the set."""
        # Raise TypeError if value is not hashable
        hash(value)

        self.redis.sadd(self.key, self._pickle(value))