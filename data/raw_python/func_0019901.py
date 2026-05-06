def get(self, uri, params={}):
        '''A generic method to make GET requests to the OpenDNS Investigate API
        on the given URI.
        '''
        return self._session.get(urljoin(Investigate.BASE_URL, uri),
            params=params, headers=self._auth_header, proxies=self.proxies
        )