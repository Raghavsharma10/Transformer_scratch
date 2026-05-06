def _get(self, uri, params={}):
        """
        HTTP GET function

        :param uri: REST endpoint
        :param params: optional HTTP params to pass to the endpoint
        :return: list of results (usually a list of dicts)

        Example:
            ret = cli.get('/search', params={ 'q': 'example.org' })
        """

        if not uri.startswith(self.remote):
            uri = '{}{}'.format(self.remote, uri)

        return self._make_request(uri, params)