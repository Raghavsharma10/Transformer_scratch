def pop(self, key, default=MISSING):
        """
        'D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
        If key is not found, d is returned if given, otherwise KeyError is raised'
        """
        try:
            val = self[key]
        except KeyError:
            if default is MISSING:
                raise
            val = default
        else:
            del self[key]
        return val