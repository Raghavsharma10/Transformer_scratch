def _load_config_file(self):
        """Load the configuration file into memory, returning the content.

        """
        LOGGER.info('Loading configuration from %s', self._file_path)
        if self._file_path.endswith('json'):
            config = self._load_json_config()
        else:
            config = self._load_yaml_config()
        for key, value in [(k, v) for k, v in config.items()]:
            if key.title() != key:
                config[key.title()] = value
                del config[key]
        return flatdict.FlatDict(config)