def retrieveVals(self):
        """Retrieve values for graphs."""
        lighttpdInfo = LighttpdInfo(self._host, self._port,
                                self._user, self._password, 
                                self._statuspath, self._ssl)
        stats = lighttpdInfo.getServerStats()
        if self.hasGraph('lighttpd_access'):
            self.setGraphVal('lighttpd_access', 'reqs', stats['Total Accesses'])
        if self.hasGraph('lighttpd_bytes'):
            self.setGraphVal('lighttpd_bytes', 'bytes', 
                             stats['Total kBytes'] * 1000)
        if self.hasGraph('lighttpd_servers'):
            self.setGraphVal('lighttpd_servers', 'busy', stats['BusyServers'])
            self.setGraphVal('lighttpd_servers', 'idle', stats['IdleServers'])
            self.setGraphVal('lighttpd_servers', 'max', stats['MaxServers'])