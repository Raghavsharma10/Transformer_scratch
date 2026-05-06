def _post(self, uri, data):
        """
        HTTP POST function

        :param uri: REST endpoint to POST to
        :param data: list of dicts to be passed to the endpoint
        :return: list of dicts, usually will be a list of objects or id's

        Example:
            ret = cli.post('/indicators', { 'indicator': 'example.com' })
        """

        if not uri.startswith(self.remote):
            uri = '{}/{}'.format(self.remote, uri)
            self.logger.debug(uri)

        return self._make_request(uri, data=data)