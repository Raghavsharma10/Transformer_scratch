def set_filters(self, filters):
        """
        set and validate filters dict
        """
        if not isinstance(filters, dict):
            raise Exception("filters must be a dict")
        self.filters = {}
        for key in filters.keys():
            value = filters[key]
            self.add_filter(key,value)