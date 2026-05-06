def _connect(self):
        """Connect to FreeSWITCH ESL Interface."""
        try:
            self._eslconn = ESL.ESLconnection(self._eslhost, 
                                              str(self._eslport), 
                                              self._eslpass)
        except:
            pass
        if not self._eslconn.connected():
            raise Exception(
                "Connection to FreeSWITCH ESL Interface on host %s and port %d failed."
                % (self._eslhost, self._eslport)
                )