def delete_category(self, category):
        """Removes the category from Redis. This doesn't touch the metrics;
        they simply become uncategorized."""
        # Remove mapping of metrics-to-category
        category_key = self._category_key(category)
        self.r.delete(category_key)

        # Remove category from Set
        self.r.srem(self._categories_key, category)