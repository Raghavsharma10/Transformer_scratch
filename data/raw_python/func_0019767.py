def _connect(self):
        """Connect to Memcached."""
        if self._socketFile is not None:
            if not os.path.exists(self._socketFile):
                raise Exception("Socket file (%s) for Memcached Instance not found."
                                % self._socketFile)
        try:
            if self._timeout is not None:
                self._conn = util.Telnet(self._host, self._port, self._socketFile, 
                                         timeout)
            else:
                self._conn = util.Telnet(self._host, self._port, self._socketFile)
        except:     
            raise Exception("Connection to %s failed." % self._instanceName)