def status(self, start, end, sources=None):
        """ Check the data coverage in the Historics archive for a given interval.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsstatus

            :param start: Unix timestamp for the start time
            :type start: int
            :param end: Unix timestamp for the start time
            :type end: int
            :param sources: list of data sources to include.
            :type sources: list
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'start': start, 'end': end}
        if sources:
            params['sources'] = ','.join(sources)
        return self.request.get('status', params=params)