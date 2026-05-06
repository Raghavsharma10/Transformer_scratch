def retrieveVals(self):
        """Retrieve values for graphs."""
        apacheInfo = ApacheInfo(self._host, self._port,
                                self._user, self._password, 
                                self._statuspath, self._ssl)
        stats = apacheInfo.getServerStats()
        if self.hasGraph('apache_access'):
            self.setGraphVal('apache_access', 'reqs', stats['Total Accesses'])
        if self.hasGraph('apache_bytes'):
            self.setGraphVal('apache_bytes', 'bytes', 
                             stats['Total kBytes'] * 1000)
        if self.hasGraph('apache_workers'):
            self.setGraphVal('apache_workers', 'busy', stats['BusyWorkers'])
            self.setGraphVal('apache_workers', 'idle', stats['IdleWorkers'])
            self.setGraphVal('apache_workers', 'max', stats['MaxWorkers'])