def _init_file(self):
        """
        Initialize a new password file and set the reference password.
        """
        self.keyring_key = self._get_new_password()
        # set a reference password, used to check that the password provided
        #  matches for subsequent checks.
        self.set_password('keyring-setting',
                          'password reference',
                          'password reference value')
        self._write_config_value('keyring-setting',
                                 'scheme',
                                 self.scheme)
        self._write_config_value('keyring-setting',
                                 'version',
                                 self.version)