def get_search_fields(self, exclude=None):
        """
        Get the fields for searching for an item.
        """
        exclude = set(exclude)
        if self.search_fields and len(self.search_fields) > 1:
            exclude = exclude.union(self.search_fields)

        return self.get_filter_fields(exclude=exclude)