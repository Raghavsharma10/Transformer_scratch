def _connect(self):
        """Connect to Squid Proxy Manager interface."""
        if sys.version_info[:2] < (2,6):
            self._conn = httplib.HTTPConnection(self._host, self._port)
        else:
            self._conn = httplib.HTTPConnection(self._host, self._port, 
                                                False, defaultTimeout)