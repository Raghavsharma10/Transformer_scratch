def request(self, method, endpoint, **kwargs):
        """
        Proxy to requests.request
        :param method: str formatted http method
        :param endpoint: service endpoint
        :param kwargs: kwargs passed directly to requests.request
        :return:
        """
        kwargs.setdefault('timeout', (1, 30))
        return self.session.request(
            method,
            urljoin(self.base_url, endpoint),
            **kwargs
        )