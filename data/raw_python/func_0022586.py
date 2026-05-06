def metric_slugs_by_category(self):
        """Return a dictionary of metrics data indexed by category:

            {<category_name>: set(<slug1>, <slug2>, ...)}

        """
        result = OrderedDict()
        categories = sorted(self.r.smembers(self._categories_key))
        for category in categories:
            result[category] = self._category_slugs(category)

        # We also need to see the uncategorized metric slugs, so need some way
        # to check which slugs are not already stored.
        categorized_metrics = set([  # Flatten the list of metrics
            slug for sublist in result.values() for slug in sublist
        ])
        f = lambda slug: slug not in categorized_metrics
        uncategorized = list(set(filter(f, self.metric_slugs())))
        if len(uncategorized) > 0:
            result['Uncategorized'] = uncategorized
        return result