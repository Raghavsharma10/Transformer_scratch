def and_filter(self, filter_or_string, *args, **kwargs):
        """
        Convenience method to delegate to the root_filter to generate an
        :class:`~es_fluent.filters.core.And` clause.

        :return: :class:`~es_fluent.builder.QueryBuilder`
        """
        self.root_filter.and_filter(filter_or_string, *args, **kwargs)
        return self