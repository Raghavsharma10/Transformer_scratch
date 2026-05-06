def connectionMade(self):
        """Register with stomp server.
        """
        cmd = stomper.connect(self.username, self.password)
        self.transport.write(cmd)