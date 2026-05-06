def add_key(self, key, first=False):
        """Adds the given key to this row.

        :param key: Key to be added to this row.
        :param first: BOolean flag that indicates if key is added at the beginning or at the end.
        """
        if first:
            self.keys = [key] + self.keys
        else:
            self.keys.append(key)
        if isinstance(key, VSpaceKey):
            self.space = key