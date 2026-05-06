def parse(self, key, value):
        """Parse the environment value for a given key against the schema.

        Args:
            key: The name of the environment variable.
            value: The value to be parsed.
        """
        if value is not None:
            try:
                return self._parser(value)
            except Exception:
                raise ParsingError("Error parsing {}".format(key))
        elif self._default is not SENTINAL:
            return self._default
        else:
            raise KeyError(key)