def _connect(self):
        """Connect to Asterisk Manager Interface."""
        try:
            if sys.version_info[:2] >= (2,6):
                self._conn = telnetlib.Telnet(self._amihost, self._amiport, 
                                              connTimeout)
            else:
                self._conn = telnetlib.Telnet(self._amihost, self._amiport)
        except:
            raise Exception(
                "Connection to Asterisk Manager Interface on "
                "host %s and port %s failed."
                % (self._amihost, self._amiport)
                )