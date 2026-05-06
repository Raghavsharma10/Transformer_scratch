def get_password(self, service, username):
        """
        Read the password from the file.
        """
        assoc = self._generate_assoc(service, username)
        service = escape_for_ini(service)
        username = escape_for_ini(username)

        # load the passwords from the file
        config = configparser.RawConfigParser()
        if os.path.exists(self.file_path):
            config.read(self.file_path)

        # fetch the password
        try:
            password_base64 = config.get(service, username).encode()
            # decode with base64
            password_encrypted = decodebytes(password_base64)
            # decrypt the password with associated data
            try:
                password = self.decrypt(password_encrypted, assoc).decode(
                    'utf-8')
            except ValueError:
                # decrypt the password without associated data
                password = self.decrypt(password_encrypted).decode('utf-8')
        except (configparser.NoOptionError, configparser.NoSectionError):
            password = None
        return password