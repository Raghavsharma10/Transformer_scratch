def get_session_value(self, name, default=None):
        """Gets a session value"""

        value = self.redis().hget(self._session_key, name) or default
        self._update_session_expiration()
        return value