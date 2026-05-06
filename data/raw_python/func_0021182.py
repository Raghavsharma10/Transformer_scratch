def client(self):
        """Return client for current application."""
        if self._client is None:
            self._client = self._client_builder()
        return self._client