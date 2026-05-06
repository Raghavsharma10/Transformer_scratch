def set_password(self, service, username, password):
        """Write the password in the file.
        """
        service = escape_for_ini(service)
        username = escape_for_ini(username)

        # encrypt the password
        password = password or ''
        password_encrypted = self.encrypt(password.encode('utf-8'))

        # encode with base64
        password_base64 = base64.encodestring(password_encrypted).decode()
        # write the modification
        if not self.config.has_section(service):
            self.config.add_section(service)
        self.config.set(service, username, password_base64)
        config_file = UnicodeWriterAdapter(self._open('w'))
        self.config.write(config_file)
        config_file.close()