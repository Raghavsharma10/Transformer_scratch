def server_bind(self):
        """Bind server to socket. Overrided to store server name and
        set timeout.
        """

        try:
            HTTPServer.server_bind(self)
        except Exception as e:
            raise WSGIServerException(e)

        self.setup_environ()
        self.socket.settimeout(1)