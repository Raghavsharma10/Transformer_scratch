def publish(self, tag, message):
        """ Publish a message down the socket """
        payload = self.build_payload(tag, message)
        self.socket.send(payload)