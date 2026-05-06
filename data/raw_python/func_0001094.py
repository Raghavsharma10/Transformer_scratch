def get(self, var, default=None):
        """Return a value from configuration.

        Safe version which always returns a default value if the value is not
        found.
        """
        try:
            return self.__get(var)
        except (KeyError, IndexError):
            return default