def get(self, id, service='facebook'):
        """ Get the existing analysis for a given hash

            :param service: The service for this API call (facebook, etc)
            :type service: str
            :param id: The optional hash to get recordings with
            :type id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'id': id}

        return self.request.get(service + '/get', params)