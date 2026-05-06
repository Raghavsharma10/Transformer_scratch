def delete_password(self, service, username):
        """Delete the password for the username of the service.
        """
        service = escape_for_ini(service)
        username = escape_for_ini(username)
        config = configparser.RawConfigParser()
        if os.path.exists(self.file_path):
            config.read(self.file_path)
        try:
            if not config.remove_option(service, username):
                raise PasswordDeleteError("Password not found")
        except configparser.NoSectionError:
            raise PasswordDeleteError("Password not found")
        # update the file
        with open(self.file_path, 'w') as config_file:
            config.write(config_file)