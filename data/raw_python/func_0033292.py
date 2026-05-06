def clear_session_value(self, name):
        """Removes a session value"""
        self.redis().hdel(self._session_key, name)
        self._update_session_expiration()