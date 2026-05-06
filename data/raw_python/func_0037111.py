def sample(self, id, count=None, start=None, end=None, filter=None,
               service='facebook'):
        """ Get sample interactions for a given hash

            :param id: The hash to get tag analysis for
            :type id: str
            :param start: Determines time period of the sample data
            :type start: int
            :param end: Determines time period of the sample data
            :type end: int
            :param filter: An optional secondary filter
            :type filter: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'id': id}

        if count:
            params['count'] = count
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        if filter:
            params['filter'] = filter

        return self.request.get(service + '/sample', params)