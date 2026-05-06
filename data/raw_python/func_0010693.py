def from_file(cls, filename, *, strict=True):
        """ Create a new Config object from a configuration file.

        Args:
            filename (str): The location and name of the configuration file.
            strict (bool): If true raises a ConfigLoadError when the configuration
                cannot be found.

        Returns:
            An instance of the Config class.

        Raises:
            ConfigLoadError: If the configuration cannot be found.
        """
        config = cls()
        config.load_from_file(filename, strict=strict)
        return config