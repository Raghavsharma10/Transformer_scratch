def send_token(self, token):
        """
        Used by `Client.connect()`.

        After connecting to an official server and sending the
        handshake packets, the client has to send the token
        acquired through `utils.find_server()`, otherwise the server will
        drop the connection when receiving any other packet.
        """
        self.send_struct('<B%iB' % len(token), 80, *map(ord, token))
        self.server_token = token