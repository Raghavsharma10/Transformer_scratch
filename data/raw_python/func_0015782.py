def get_indexes(self, default_indexes=DEFAULT_INDEXES):
        """Returns the list of indexes to act on."""
        for action, value in reversed(self.steps):
            if action == 'indexes':
                return list(value)

        if self.type is not None:
            indexes = self.type.get_index()
            if isinstance(indexes, string_types):
                indexes = [indexes]
            return indexes

        return default_indexes