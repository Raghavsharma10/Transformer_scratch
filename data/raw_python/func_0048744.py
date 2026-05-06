def build_filtered_queryset(self, query, **kwargs):
        """
        Build and return the fully-filtered queryset
        """
        # Take the basic queryset
        qs = self.get_queryset()
        # filter it via the query conditions
        qs = qs.filter(self.get_queryset_filters(query))
        return self.build_extra_filtered_queryset(qs, **kwargs)