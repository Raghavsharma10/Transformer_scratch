def or_filter(self, filter_or_string, *args, **kwargs):
        """
        Convenience method to delegate to the root_filter to generate an `or`
        clause.

        :return: :class:`~es_fluent.builder.QueryBuilder`
        """
        self.root_filter.or_filter(filter_or_string, *args, **kwargs)
        return self