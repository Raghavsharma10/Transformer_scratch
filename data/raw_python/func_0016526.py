def close(self):
        """Close the channel to the queue."""
        self.cancel()
        self.backend.close()
        self._closed = True