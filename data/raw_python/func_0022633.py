def get_context_data(self, **kwargs):
        """Includes the Gauge slugs and data in the context."""
        data = super(GaugesView, self).get_context_data(**kwargs)
        data.update({'gauges': get_r().gauge_slugs()})
        return data