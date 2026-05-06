def enable_ssl(self, *args, **kwargs):
        """
        Transforms the regular socket.socket to an ssl.SSLSocket for secure
        connections. Any arguments are passed to ssl.wrap_socket:
        http://docs.python.org/dev/library/ssl.html#ssl.wrap_socket
        """
        if self.handshake_sent:
            raise SSLError('can only enable SSL before handshake')

        self.secure = True
        self.sock = ssl.wrap_socket(self.sock, *args, **kwargs)