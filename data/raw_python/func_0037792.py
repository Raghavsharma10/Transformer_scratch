def filter(self, collection, data, **kwargs):
        """Filter given collection."""
        ops = self.parse(data)
        collection = self.apply(collection, ops, **kwargs)
        return ops, collection