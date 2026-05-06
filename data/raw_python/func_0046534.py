def or_filter(self, filter_or_string, *args, **kwargs):
        """
        Adds a list of :class:`~es_fluent.filters.core.Or` clauses, automatically
        generating the an :class:`~es_fluent.filters.core.Or` filter if it does not
        exist.
        """
        or_filter = self.find_filter(Or)

        if or_filter is None:
            or_filter = Or()
            self.filters.append(or_filter)

        or_filter.add_filter(build_filter(
            filter_or_string, *args, **kwargs
        ))

        return or_filter