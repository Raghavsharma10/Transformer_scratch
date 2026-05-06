def initStats(self):
        """Query and parse Apache Web Server Status Page."""
        url = "%s://%s:%d/%s?auto"  % (self._proto, self._host, self._port, 
                                       self._statuspath)
        response = util.get_url(url, self._user, self._password)
        self._statusDict = {}
        for line in response.splitlines():
            mobj = re.match('(\S.*\S)\s*:\s*(\S+)\s*$', line)
            if mobj:
                self._statusDict[mobj.group(1)] = util.parse_value(mobj.group(2))
        if self._statusDict.has_key('Scoreboard'):
            self._statusDict['MaxWorkers'] = len(self._statusDict['Scoreboard'])