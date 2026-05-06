def auth(self):
        """
        tuple of (username, password). if use_keyring is set to true the password will be queried from the local keyring instead of taken from the
        configuration file.
        """
        username = self._settings["username"]

        if not username:
            raise ValueError("Username was not configured in %s" % CONFIG_FILE)

        if self._settings["use_keyring"]:
            password = self.keyring_get_password(username)
            if not password:
                self.keyring_set_password(username)
                password = self.keyring_get_password(username)
        else:
            password = self._settings["password"]

        return self._settings["username"], password