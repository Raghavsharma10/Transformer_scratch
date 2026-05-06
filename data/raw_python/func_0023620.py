def write_config(self):
        """Write the current configuration to the config file."""
        config_file = os.path.join(self.config_dir, 'pueue.ini')
        with open(config_file, 'w') as file_descriptor:
            self.config.write(file_descriptor)