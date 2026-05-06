def tags(self, id, service='facebook'):
        """ Get the existing analysis for a given hash

            :param id: The hash to get tag analysis for
            :type id: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """
        return self.request.get(service + '/tags', params=dict(id=id))