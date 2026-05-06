def close(self):
        """Shut down the socket connection, client and controller"""
        self._sock = None
        self._controller = None
        if hasattr(self, "_port") and self._port:
            portpicker.return_port(self._port)
            self._port = None