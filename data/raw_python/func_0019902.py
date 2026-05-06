def post(self, uri, params={}, data={}):
        '''A generic method to make POST requests to the OpenDNS Investigate API
        on the given URI.
        '''
        return self._session.post(
            urljoin(Investigate.BASE_URL, uri),
            params=params, data=data, headers=self._auth_header,
            proxies=self.proxies
        )