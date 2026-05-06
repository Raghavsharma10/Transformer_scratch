def send(self, data):
        """Send a formatted message to the ADB server"""
        self._send_data(int_to_hex(len(data)))

        self._send_data(data)