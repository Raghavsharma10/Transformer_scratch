def payload(self):
        """The decoded message."""
        if not self._decoded_cache:
            self._decoded_cache = self.decode()
        return self._decoded_cache