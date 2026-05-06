def timeout(self, value):
        """Sets a custom timeout value for this session"""

        if value == TIMEOUT_SESSION:
            self._config.timeout = None
            self._backend_client.expires = None
        else:
            self._config.timeout = value
            self._calculate_expires()