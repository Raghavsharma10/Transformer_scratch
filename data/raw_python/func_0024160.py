def add(self, val):
        """
        Add given value to item (list)

        Args:
            val: A JSON serializable object.

        Returns:
            Cache backend response.
        """
        return cache.lpush(self.key, json.dumps(val) if self.serialize else val)