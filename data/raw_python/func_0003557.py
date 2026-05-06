def send(self):
        """Send this message to the controller."""
        self._file.write(self.as_bytes())
        self._file.write(b'\r\n')