def apply_published_filter(self, queryset, operation, value):
        """
        Add the appropriate Published filter to a given elasticsearch query.

        :param queryset: The DJES queryset object to be filtered.
        :param operation: The type of filter (before/after).
        :param value: The date or datetime value being applied to the filter.
        """
        if operation not in ["after", "before"]:
            raise ValueError("""Publish filters only use before or after for range filters.""")
        return queryset.filter(Published(**{operation: value}))