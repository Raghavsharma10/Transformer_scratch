def _check_file(self):
        """
        Check if the file exists and has the expected password reference.
        """
        if not os.path.exists(self.file_path):
            return False
        self._migrate()
        config = configparser.RawConfigParser()
        config.read(self.file_path)
        try:
            config.get(
                escape_for_ini('keyring-setting'),
                escape_for_ini('password reference'),
            )
        except (configparser.NoSectionError, configparser.NoOptionError):
            return False
        try:
            self._check_scheme(config)
        except AttributeError:
            # accept a missing scheme
            return True
        return self._check_version(config)