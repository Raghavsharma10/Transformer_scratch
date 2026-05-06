def reset_category(self, category, metric_slugs):
        """Resets (or creates) a category containing a list of metrics.

        * ``category`` -- A category name
        * ``metric_slugs`` -- a list of all metrics that are members of the
            category.

        """
        key = self._category_key(category)
        if len(metric_slugs) == 0:
            # If there are no metrics, just remove the category
            self.delete_category(category)
        else:
            # Save all the slugs in the category, and save the category name
            self.r.sadd(key, *metric_slugs)
            self.r.sadd(self._categories_key, category)