def filter(self, data, collection, **kwargs):
        """Filter given collection."""
        if not data or self.filters is None:
            return None, collection

        filters = {}
        for f in self.filters:
            if f.name not in data:
                continue
            ops, collection = f.filter(collection, data, **kwargs)
            filters[f.name] = ops

        return filters, collection