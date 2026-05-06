def set_session_value(self, name, value):
        """Sets a session value"""

        self.redis().hset(self._session_key, name, value)
        self._update_session_expiration()