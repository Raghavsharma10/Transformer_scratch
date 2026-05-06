def hostname(self, value):
        """
        The hostname where the log message was created.

        Should be the first part of the hostname, or
        an IP address. Should NOT be set to a fully
        qualified domain name.

        """
        if value is None:
            value = socket.gethostname()
        self._hostname = value