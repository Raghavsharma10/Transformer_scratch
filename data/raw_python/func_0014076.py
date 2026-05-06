def open(self):
        """Open the connection wit the device."""
        try:
            self.device.open()
        except ConnectTimeoutError as cte:
            raise ConnectionException(cte.message)
        self.device.timeout = self.timeout
        self.device._conn._session.transport.set_keepalive(self.keepalive)
        if hasattr(self.device, "cu"):
            # make sure to remove the cu attr from previous session
            # ValueError: requested attribute name cu already exists
            del self.device.cu
        self.device.bind(cu=Config)
        if self.config_lock:
            self._lock()