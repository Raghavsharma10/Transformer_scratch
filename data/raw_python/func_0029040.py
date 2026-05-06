def reload(self):
        """Reload the configuration from disk returning True if the
        configuration has changed from the previous values.

        """
        config = self._default_configuration()
        if self._file_path:
            config.update(self._load_config_file())
        if config != self._values:
            self._values = config
            return True
        return False