def filter_queryset(self, request, queryset, view):
        """Apply the relevant behaviors to the view queryset."""
        start_value = self.get_start(request)
        if start_value:
            queryset = self.apply_published_filter(queryset, "after", start_value)
        end_value = self.get_end(request)
        if end_value:
            # Forces the end_value to be the last second of the date provided in the query.
            # Necessary currently as our Published filter for es only applies to gte & lte.
            queryset = self.apply_published_filter(queryset, "before", end_value)
        return queryset