def find_filter(self, filter_cls):
        """
        Find or create a filter instance of the provided ``filter_cls``. If it
        is found, use remaining arguments to augment the filter otherwise
        create a new instance of the desired type and add it to the
        current :class:`~es_fluent.builder.QueryBuilder` accordingly.
        """
        for filter_instance in self.filters:
            if isinstance(filter_instance, filter_cls):
                return filter_instance

        return None