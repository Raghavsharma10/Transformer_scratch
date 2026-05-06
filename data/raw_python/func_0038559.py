def disconnect(self):
        """ Disconnect from chassis and server. """
        if self.root.ref is not None:
            self.api.disconnect()
        self.root = None