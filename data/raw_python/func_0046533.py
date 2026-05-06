def and_filter(self, filter_or_string, *args, **kwargs):
        """
        Adds a list of :class:`~es_fluent.filters.core.And` clauses, automatically
        generating :class:`~es_fluent.filters.core.And` filter if it does not
        exist.
        """
        and_filter = self.find_filter(And)

        if and_filter is None:
            and_filter = And()
            self.filters.append(and_filter)

        and_filter.add_filter(build_filter(
            filter_or_string, *args, **kwargs))

        return and_filter