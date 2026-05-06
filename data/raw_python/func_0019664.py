def getStats(self):
        """Query and parse Web Server Status Page.
        
        """
        url = "%s://%s:%d/%s" % (self._proto, self._host, self._port, 
                                 self._monpath)
        response = util.get_url(url, self._user, self._password)
        stats = {}
        for line in response.splitlines():
            mobj = re.match('([\w\s]+):\s+(\w+)$', line)
            if mobj:
                stats[mobj.group(1)] = util.parse_value(mobj.group(2))
        return stats