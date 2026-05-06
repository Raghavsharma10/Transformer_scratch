def set(self, key, value):
        """Set a value in the `Bison` configuration.

        Args:
            key (str): The configuration key to set a new value for.
            value: The value to set.
        """
        # the configuration changes, so we invalidate the cached config
        self._full_config = None
        self._override[key] = value