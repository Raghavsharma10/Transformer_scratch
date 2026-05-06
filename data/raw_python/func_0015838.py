def get_password(self, service, username):
        """Read the password from the file.
        """
        service = escape_for_ini(service)
        username = escape_for_ini(username)

        # fetch the password
        try:
            password_base64 = self.config.get(service, username).encode()
            # decode with base64
            password_encrypted = base64.decodestring(password_base64)
            # decrypted the password
            password = self.decrypt(password_encrypted).decode('utf-8')
        except (configparser.NoOptionError, configparser.NoSectionError):
            password = None
        return password