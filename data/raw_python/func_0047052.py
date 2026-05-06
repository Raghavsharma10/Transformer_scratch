def save_config(self, config_path=None):
        """Save configuration to the specified path, or self.config_path"""
        if config_path is None:
            config_path = self.config_path
        else:
            self.config_path = config_path

        with open(config_path, 'w') as f:
            self.config.write(f)