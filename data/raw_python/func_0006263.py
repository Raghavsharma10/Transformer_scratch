def login(self, login, password):
        """
            Login to the remote telnet server.

        :param login: Username to use for logging in
        :param password: Password to use for logging in
        :raise: `InvalidLogin` on failed login
        """
        self.client.read_until('Username: ')
        self.client.write(login + '\r\n')
        self.client.read_until('Password: ')
        self.client.write(password + '\r\n')
        current_data = self.client.read_until('$ ', 10)
        if not current_data.endswith('$ '):
            raise InvalidLogin