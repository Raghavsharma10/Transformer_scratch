def execute(self, method, path, **kwargs):
        """ Executes a request to a given endpoint, returning the result """

        url = "{}{}".format(self.host, path)
        kwargs.update(self._client_kwargs)
        response = requests.request(
            method,
            url,
            headers={"Authorization": "Bearer {}".format(self.api_key)},
            **kwargs)
        return response