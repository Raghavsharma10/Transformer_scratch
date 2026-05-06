def initStats(self, extras=None):
        """Query and parse Web Server Status Page.
        
        @param extras: Include extra metrics, which can be computationally more 
                       expensive.
        
        """
        url = "%s://%s:%d/%s" % (self._proto, self._host, self._port, self._monpath)
        response = util.get_url(url, self._user, self._password)
        #with open('/tmp/opcinfo.json') as f:
        #    response = f.read()
        self._statusDict = json.loads(response)