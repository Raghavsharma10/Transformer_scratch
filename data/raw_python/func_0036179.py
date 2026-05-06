def check_correct_host(self):
        """The method checks whether the host is correctly set and whether it can configure the connection to this host. Does not check the base host knoema.com """
        if self._host == 'knoema.com':
            return
        url = self._get_url('/api/1.0/frontend/tags')
        req = urllib.request.Request(url)
        try:
            resp = urllib.request.urlopen(req)
        except:
            raise ValueError('The specified host {} does not exist'.format(self._host))