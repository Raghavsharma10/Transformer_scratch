def get_metric_history(self, slugs, since=None, to=None, granularity='daily'):
        """Get history for one or more metrics.

        * ``slugs`` -- a slug OR a list of slugs
        * ``since`` -- the date from which we start pulling metrics
        * ``to`` -- the date until which we start pulling metrics
        * ``granularity`` -- seconds, minutes, hourly,
                             daily, weekly, monthly, yearly

        Returns a list of tuples containing the Redis key and the associated
        metric::

            r = R()
            r.get_metric_history('test', granularity='weekly')
            [
                ('m:test:w:2012-52', '15'),
            ]

        To get history for multiple metrics, just provide a list of slugs::

            metrics = ['test', 'other']
            r.get_metric_history(metrics, granularity='weekly')
            [
                ('m:test:w:2012-52', '15'),
                ('m:other:w:2012-52', '42'),
            ]

        """
        if not type(slugs) == list:
            slugs = [slugs]

        # Build the set of Redis keys that we need to get.
        keys = []
        for slug in slugs:
            for date in self._date_range(granularity, since, to):
                keys += self._build_keys(slug, date, granularity)
        keys = list(dedupe(keys))

        # Fetch our data, replacing any None-values with zeros
        results = [0 if v is None else v for v in self.r.mget(keys)]
        results = zip(keys, results)
        return sorted(results, key=lambda t: t[0])