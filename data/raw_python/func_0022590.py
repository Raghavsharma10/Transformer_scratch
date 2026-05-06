def get_metric(self, slug):
        """Get the current values for a metric.

        Returns a dictionary with metric values accumulated for the seconds,
        minutes, hours, day, week, month, and year.

        """
        results = OrderedDict()
        granularities = self._granularities()
        keys = self._build_keys(slug)
        for granularity, key in zip(granularities, keys):
            results[granularity] = self.r.get(key)
        return results