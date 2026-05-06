def get_category_metrics(self, category):
        """Get metrics belonging to the given category"""
        slug_list = self._category_slugs(category)
        return self.get_metrics(slug_list)