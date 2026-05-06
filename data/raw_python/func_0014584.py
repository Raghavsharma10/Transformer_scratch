def chassis(self):
        """Get list of chassis known to test session."""
        self._check_session()
        status, data = self._rest.get_request('chassis')
        return data