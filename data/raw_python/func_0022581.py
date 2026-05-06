def _category_slugs(self, category):
        """Returns a set of the metric slugs for the given category"""
        key = self._category_key(category)
        slugs = self.r.smembers(key)
        return slugs