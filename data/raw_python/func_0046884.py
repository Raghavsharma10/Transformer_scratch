def _get_xml(self, metric):
        """Returns the channel element of the RSS feed"""
        self._opener = urllib2.build_opener()
        self._opener.addheaders = [('User-agent', self.user_agent)]

        if metric:
            url = self.base_url + '?w={0}&u=c'.format(self.woeid)
        else:
            url = self.base_url + '?w={0}'.format(self.woeid)

        return etree.parse(
            self._opener.open(url)
        ).getroot()[0]