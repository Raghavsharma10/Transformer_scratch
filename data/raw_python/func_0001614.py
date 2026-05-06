def send(self, payload):
        """ Encode and sign (optional) the send through socket """
        payload = self.encode(payload)
        payload = self.sign(payload)
        self.socket.send(payload)