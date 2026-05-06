def read_settings(self):
        """
        Read the "dsbfile" file
        Populates `self.settings`
        """
        logger.debug('Reading settings from: %s', self.settings_path)
        self.settings = Settings.from_dsbfile(self.settings_path)