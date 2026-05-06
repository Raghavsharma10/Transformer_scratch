def remove_item(self, val):
        """
        Removes given item from the list.

        Args:
            val: Item

        Returns:
            Cache backend response.
        """
        return cache.lrem(self.key, json.dumps(val))