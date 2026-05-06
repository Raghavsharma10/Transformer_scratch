def get_context_data(self, **kwargs):
        """Includes the metrics slugs in the context."""
        r = get_r()
        data = super(AggregateHistoryView, self).get_context_data(**kwargs)
        slug_set = set(kwargs['slugs'].split('+'))
        granularity = kwargs.get('granularity', 'daily')

        # Accept GET query params for ``since``
        since = self.request.GET.get('since', None)
        if since and len(since) == 10:  # yyyy-mm-dd
            since = datetime.strptime(since, "%Y-%m-%d")
        elif since and len(since) == 19:  # yyyy-mm-dd HH:MM:ss
            since = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")

        data.update({
            'slugs': slug_set,
            'granularity': granularity,
            'since': since,
            'granularities': list(r._granularities())
        })
        return data