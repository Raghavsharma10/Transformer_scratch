def connect(self):
        """Connect to the given socket"""
        if not self.connected:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect(self.address)
            self.connected = True