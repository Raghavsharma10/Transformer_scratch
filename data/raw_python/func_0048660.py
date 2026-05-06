def get(self, key, default=None, type_=None):
        """
        Return the last data value for the passed key. If key doesn't exist
        or value is an empty list, return `default`.
        """
        try:
            rv = self[key]
        except KeyError:
            return default
        if type_ is not None:
            try:
                rv = type_(rv)
            except ValueError:
                rv = default
        return rv