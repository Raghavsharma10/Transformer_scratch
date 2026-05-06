def authenticate(self, username, password):
        """
        Authenticate user on server.

        :param username: Username used to be authenticated.
        :type username: six.string_types
        :param password: Password used to be authenticated.
        :type password: six.string_types
        :return: True if successful.
        :raises: InvalidCredentials, AuthenticationNotSupported, MemcachedException
        :rtype: bool
        """
        self._username = username
        self._password = password

        # Reopen the connection with the new credentials.
        self.disconnect()
        self._open_connection()
        return self.authenticated