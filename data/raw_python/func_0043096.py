def terminate(self):
        """Terminates an active session"""
        self._backend_client.clear()
        self._needs_save = False
        self._started = False
        self._expire_cookie = True
        self._send_cookie = True