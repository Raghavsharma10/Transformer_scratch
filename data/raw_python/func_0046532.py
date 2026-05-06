def add_filter(self, filter_or_string, *args, **kwargs):
        """
        Appends a filter.
        """
        self.filters.append(build_filter(filter_or_string, *args, **kwargs))

        return self