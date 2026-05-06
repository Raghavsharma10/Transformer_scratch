def add_filter(self, filter_or_string, *args, **kwargs):
        """
        Adds a filter to the query builder's filters.

        :return: :class:`~es_fluent.builder.QueryBuilder`
        """
        self.root_filter.add_filter(filter_or_string, *args, **kwargs)
        return self