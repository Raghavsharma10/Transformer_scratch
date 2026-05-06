def get(self, key):
        """Retrieve a value from the session dictionary"""
        self._started = self._backend_client.load()
        self._needs_save = True

        return self._backend_client.get(key)