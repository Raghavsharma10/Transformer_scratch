def receive(self, decode=True):
        """ Receive from socket, authenticate and decode payload """
        payload = self.socket.recv()
        payload = self.verify(payload)
        if decode:
            payload = self.decode(payload)
        return payload