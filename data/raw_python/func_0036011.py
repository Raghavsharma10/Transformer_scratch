def update(self, mapping):
        """Update database with the key/values from a :class:`dict`."""
        return self.api.mset(dict((key, self.prepare_value(value))
                                for key, value in mapping.items()))