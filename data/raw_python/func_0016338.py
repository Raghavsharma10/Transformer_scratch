def update_kwargs(self, kwargs, count, offset):
        """
        Helper to support handy dictionaries merging on all Python versions.
        """
        kwargs.update({self.count_key: count, self.offset_key: offset})
        return kwargs