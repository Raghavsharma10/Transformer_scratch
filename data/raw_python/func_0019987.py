def convert_mapping(self, mapping):
        """Convenience method to track either a dict or a 2-tuple iterator."""
        if isinstance(mapping, dict):
            return self.convert_items(iteritems(mapping))
        return self.convert_items(mapping)