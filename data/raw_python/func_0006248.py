def connect_login(self):
        """
            Try to login to the Remote SSH Server.

        :return: Response text on successful login
        :raise: `AuthenticationFailed` on unsuccessful login
        """
        self.client.connect(self.options['server'], self.options['port'], self.options['username'],
                            self.options['password'])
        self.comm_chan = self.client.invoke_shell()
        time.sleep(1)  # Let the server take some time to get ready.
        while not self.comm_chan.recv_ready():
            time.sleep(0.5)
        login_response = self.comm_chan.recv(2048)
        if not login_response.endswith('$ '):
            raise AuthenticationFailed
        return login_response