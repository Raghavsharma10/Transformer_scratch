def set_password(self, service, username, password):
        """Write the password in the file.
        """
        if not username:
            # https://github.com/jaraco/keyrings.alt/issues/21
            raise ValueError("Username cannot be blank.")
        if not isinstance(password, string_types):
            raise TypeError("Password should be a unicode string, not bytes.")
        assoc = self._generate_assoc(service, username)
        # encrypt the password
        password_encrypted = self.encrypt(password.encode('utf-8'), assoc)
        # encode with base64 and add line break to untangle config file
        password_base64 = '\n' + encodebytes(password_encrypted).decode()

        self._write_config_value(service, username, password_base64)