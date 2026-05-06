def start(self, hash, name=None, service='facebook'):
        """ Start a recording for the provided hash

            :param hash: The hash to start recording with
            :type hash: str
            :param name: The name of the recording
            :type name: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'hash': hash}

        if name:
            params['name'] = name

        return self.request.post(service + '/start', params)