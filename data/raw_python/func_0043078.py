def config(self):
        """Get the complete configuration where the default, config,
        environment, and override values are merged together.

        Returns:
            (DotDict): A dictionary of configuration values that
                allows lookups using dot notation.
        """
        if self._full_config is None:
            self._full_config = DotDict()
            self._full_config.merge(self._default)
            self._full_config.merge(self._config)
            self._full_config.merge(self._environment)
            self._full_config.merge(self._override)
        return self._full_config