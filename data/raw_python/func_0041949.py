def ensure_connected(self):
        """Ensures database connection is still open."""
        if not self.is_connected():
            if not self._auto_connect:
                raise DBALConnectionError.connection_closed()
            self.connect()