def _calculate_expires(self):
        """Calculates the session expiry using the timeout"""
        self._backend_client.expires = None

        now = datetime.utcnow()
        self._backend_client.expires = now + timedelta(seconds=self._config.timeout)