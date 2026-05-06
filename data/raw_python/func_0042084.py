def use_config_file(self):
        """Find and apply the config file"""
        self.config_file = self.find_config_file()
        if self.config_file:
            self.apply_config_file(self.config_file)