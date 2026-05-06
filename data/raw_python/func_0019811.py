def initStats(self):
        """Query and parse Nginx Web Server Status Page."""
        url = "%s://%s:%d/%s" % (self._proto, self._host, self._port, 
                                 self._statuspath)
        response = util.get_url(url, self._user, self._password)
        self._statusDict = {}
        for line in response.splitlines():
            mobj = re.match('\s*(\d+)\s+(\d+)\s+(\d+)\s*$', line)
            if mobj:
                idx = 0
                for key in ('accepts','handled','requests'):
                    idx += 1
                    self._statusDict[key] = util.parse_value(mobj.group(idx))
            else:
                for (key,val) in re.findall('(\w+):\s*(\d+)', line):
                    self._statusDict[key.lower()] = util.parse_value(val)