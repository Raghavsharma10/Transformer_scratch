def _check_version(self, config):
        """
        check for a valid version
        an existing scheme implies an existing version as well

        return True, if version is valid, and False otherwise
        """
        try:
            self.file_version = config.get(
                escape_for_ini('keyring-setting'),
                escape_for_ini('version'),
            )
        except (configparser.NoSectionError, configparser.NoOptionError):
            return False
        return True