def connect(self, address):
        """
        Equivalent to socket.connect(), but sends an client handshake request
        after connecting.

        `address` is a (host, port) tuple of the server to connect to.
        """
        self.sock.connect(address)
        ClientHandshake(self).perform()
        self.handshake_sent = True