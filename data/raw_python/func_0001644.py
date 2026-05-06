def get_client(self):
        """Gets the SSH client.

        This will check that the connection is still alive first, and reconnect
        if necessary.
        """
        if self._ssh is None:
            self._connect()
            return self._ssh
        else:
            try:
                chan = self._ssh.get_transport().open_session()
            except (socket.error, paramiko.SSHException):
                logger.warning("Lost connection, reconnecting...")
                self._ssh.close()
                self._connect()
            else:
                chan.close()
            return self._ssh