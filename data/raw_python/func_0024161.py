def get_all(self):
        """
        Get all list items.

        Returns:
            Cache backend response.
        """
        result = cache.lrange(self.key, 0, -1)
        return (json.loads(item.decode('utf-8')) for item in result if
                item) if self.serialize else result