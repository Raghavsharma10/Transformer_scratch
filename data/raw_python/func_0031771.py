def get(self, url, proto='http'):
        """
        Load an url using the GET method.

        Keyword arguments:
        url -- the Universal Resource Location
        proto -- the protocol (default 'http')
        """
        self.last_response = self.session.get(proto + self.base_uri + url,
                                              headers=self.headers,
                                              cookies=self.cookies,
                                              allow_redirects=True,
                                              verify=self.verify)
        return self.last_response_soup