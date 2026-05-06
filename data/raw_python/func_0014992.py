def _send_and_reconnect(self, message):
        """Send _message_ to Graphite Server and attempt reconnect on failure.

        If _autoreconnect_ was specified, attempt to reconnect if first send
        fails.

        :raises AttributeError: When the socket has not been set.
        :raises socket.error: When the socket connection is no longer valid.
        """
        try:
            self.socket.sendall(message.encode("ascii"))
        except (AttributeError, socket.error):
            if not self.autoreconnect():
                raise
            else:
                self.socket.sendall(message.encode("ascii"))