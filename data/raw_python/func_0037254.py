def get_connection(self):
        """Creates a new connection"""
        if self.host is None:
            connector = SpamCUnixConnector
            conn = connector(self.socket_file, self.backend_mod)
        else:
            connector = SpamCTcpConnector
            conn = connector(
                self.host,
                self.port,
                self.backend_mod,
                is_ssl=self.is_ssl,
                **self.ssl_args)
        return conn