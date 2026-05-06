def stop(self):
        """Stop the communication with the shield."""
        with self.lock:
            self._message_received(ConnectionClosed(self._file, self))