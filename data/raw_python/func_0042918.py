def get_filter_fields(self, exclude=None):
        """
        Get the fields that are normal filter fields
        """

        exclude_set = set(self.exclude)
        if exclude:
            exclude_set = exclude_set.union(set(exclude))

        return [name for name in self.fields
                if name not in exclude_set]