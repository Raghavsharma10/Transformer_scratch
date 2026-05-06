def usage(self, period=None, start=None, end=None):
        """ Get account usage information

            :param period: Period is one of either hourly, daily or monthly
            :type period: str
            :param start: Determines time period of the usage
            :type start: int
            :param end: Determines time period of the usage
            :type end: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {}

        if period:
            params['period'] = period
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        return self.request.get('usage', params)