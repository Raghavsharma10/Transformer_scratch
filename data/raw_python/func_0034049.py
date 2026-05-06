def send_struct(self, fmt, *data):
        """
        If connected, formats the data to a struct and sends it to the server.
        Used internally by all other `send_*()` methods.
        """
        if self.connected:
            self.ws.send(struct.pack(fmt, *data))