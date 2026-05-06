def initStats(self, extras=None):
        """Query and parse Web Server Status Page.
        
        @param extras: Include extra metrics, which can be computationally more 
                       expensive.
        
        """
        if extras is not None:
            self._extras = extras
        if self._extras:
            detail = 1
        else:
            detail = 0
        url = "%s://%s:%d/%s?detail=%s" % (self._proto, self._host, self._port, 
                                           self._monpath, detail)
        response = util.get_url(url, self._user, self._password)
        self._statusDict = {}
        for line in response.splitlines():
            cols = line.split(':')
            if not self._statusDict.has_key(cols[0]):
                self._statusDict[cols[0]] = {}
            self._statusDict[cols[0]][cols[1]] = util.parse_value(cols[2])