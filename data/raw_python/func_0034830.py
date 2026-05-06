def _on_closed(self):
        """Invoked when the connection is closed"""
        LOGGER.error('Redis connection closed')
        self.connected = False
        self._on_close()
        self._stream = None