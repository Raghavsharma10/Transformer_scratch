def connectionMade(self):
        """Register with the stomp server.
        """
        cmd = self.sm.connect()
        self.transport.write(cmd)