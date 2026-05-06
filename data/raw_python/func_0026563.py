def _can_connect(self):
        """Tries to connect to the configured host:port and returns True if the connection was established"""
        self.log('Trying to reach configured connectivity check endpoint', lvl=verbose)

        try:
            socket.setdefaulttimeout(self.config.timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.config.host, self.config.port))
            return True
        except Exception as ex:
            self.log(ex, pretty=True, lvl=debug)
            return False