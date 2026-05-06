def load_config(self, config_path=None):
        """Load configuration from the specified path, or self.config_path"""
        if config_path is None:
            config_path = self.config_path
        else:
            self.config_path = config_path

        config = ConfigParser.SafeConfigParser(self.DEFAULT_SUBSTITUTIONS,
                                               allow_no_value=True)
        # Avoid the configparser automatically lowercasing keys
        config.optionxform = str
        self.initialize_config(config)
        try:
            with open(config_path) as f:
                config.readfp(f)
        except (IOError, ConfigParser.Error):
            _log.exception("Ignoring config from %s due to error.", config_path)
            return False

        self.config = config
        self.reload_modules()
        return True