def analyze(self, id, parameters, filter=None, start=None, end=None,
                service='facebook'):
        """ Analyze the recorded data for a given hash

            :param id: The id of the recording
            :type id: str
            :param parameters: To set settings such as threshold and target
            :type parameters: dict
            :param filter: An optional secondary filter
            :type filter: str
            :param start: Determines time period of the analyze
            :type start: int
            :param end: Determines time period of the analyze
            :type end: int
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'id': id,
                  'parameters': parameters}

        if filter:
            params['filter'] = filter
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        return self.request.post(service + '/analyze', params)