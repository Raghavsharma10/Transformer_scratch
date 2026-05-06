def logout(self):
        """
            Logout from the remote server.
        """
        self.client.write('exit\r\n')
        self.client.read_all()
        self.client.close()