def _retrieve(self):
        """Query Apache Tomcat Server Status Page in XML format and return 
        the result as an ElementTree object.
        
        @return: ElementTree object of Status Page XML.
        
        """
        url = "%s://%s:%d/manager/status" % (self._proto, self._host, self._port)
        params = {}
        params['XML'] = 'true'
        response = util.get_url(url, self._user, self._password, params)
        tree = ElementTree.XML(response)
        return tree