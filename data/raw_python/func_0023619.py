def read_config(self):
        """Read a previous configuration file or create a new with default values."""
        config_file = os.path.join(self.config_dir, 'pueue.ini')
        self.config = configparser.ConfigParser()
        # Try to get configuration file and return it
        # If this doesn't work, a new default config file will be created
        if os.path.exists(config_file):
            try:
                self.config.read(config_file)
                return
            except Exception:
                self.logger.error('Error while parsing config file. Deleting old config')
                self.logger.exception()

        self.config['default'] = {
            'resumeAfterStart': False,
            'maxProcesses': 1,
            'customShell': 'default',
        }
        self.config['log'] = {
            'logTime': 60*60*24*14,
        }
        self.write_config()