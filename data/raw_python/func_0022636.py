def get_context_data(self, **kwargs):
        """Includes the metrics slugs in the context."""
        data = super(MetricHistoryView, self).get_context_data(**kwargs)

        # Accept GET query params for ``since``
        since = self.request.GET.get('since', None)
        if since and len(since) == 10:  # yyyy-mm-dd
            since = datetime.strptime(since, "%Y-%m-%d")
        elif since and len(since) == 19:  # yyyy-mm-dd HH:MM:ss
            since = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")

        data.update({
            'since': since,
            'slug': kwargs['slug'],
            'granularity': kwargs['granularity'],
            'granularities': list(get_r()._granularities()),
        })
        return data