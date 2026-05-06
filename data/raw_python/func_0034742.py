def accept(self):
        """
        Equivalent to socket.accept(), but transforms the socket into a
        websocket instance and sends a server handshake (after receiving a
        client handshake). Note that the handshake may raise a HandshakeError
        exception.
        """
        sock, address = self.sock.accept()
        wsock = websocket(sock)
        wsock.secure = self.secure
        ServerHandshake(wsock).perform(self)
        wsock.handshake_sent = True
        return wsock, address