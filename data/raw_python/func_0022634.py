def get_context_data(self, **kwargs):
        """Includes the metrics slugs in the context."""
        data = super(MetricsListView, self).get_context_data(**kwargs)

        # Metrics organized by category, like so:
        # { <category_name>: [ <slug1>, <slug2>, ... ]}
        data.update({'metrics': get_r().metric_slugs_by_category()})
        return data