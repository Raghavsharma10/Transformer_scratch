def get_context_data(self, **kwargs):
        """Includes the metrics slugs in the context."""
        r = get_r()
        category = kwargs.pop('category', None)
        data = super(AggregateDetailView, self).get_context_data(**kwargs)

        if category:
            slug_set =   r._category_slugs(category)
        else:
            slug_set = set(kwargs['slugs'].split('+'))

        data['granularities'] = list(r._granularities())
        data['slugs'] = slug_set
        data['category'] = category
        return data