def _cast_boolean(self, value):
        """
        Helper to convert config values to boolean as ConfigParser do.
        """
        if value.lower() not in self._BOOLEANS:
            raise ValueError('Not a boolean: %s' % value)

        return self._BOOLEANS[value.lower()]