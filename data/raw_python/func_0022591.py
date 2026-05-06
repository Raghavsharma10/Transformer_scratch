def get_metrics(self, slug_list):
        """Get the metrics for multiple slugs.

        Returns a list of two-tuples containing the metric slug and a
        dictionary like the one returned by ``get_metric``::

            (
                some-metric, {
                    'seconds': 0, 'minutes': 0, 'hours': 0,
                    'day': 0, 'week': 0, 'month': 0, 'year': 0
                }
            )

        """
        # meh. I should have been consistent here, but I'm lazy, so support these
        # value names instead of granularity names, but respect the min/max
        # granularity settings.
        keys = ['seconds', 'minutes', 'hours', 'day', 'week', 'month', 'year']
        key_mapping = {gran: key for gran, key in zip(GRANULARITIES, keys)}
        keys = [key_mapping[gran] for gran in self._granularities()]

        results = []
        for slug in slug_list:
            metrics = self.r.mget(*self._build_keys(slug))
            if any(metrics):  # Only if we have data.
                results.append((slug, dict(zip(keys, metrics))))
        return results