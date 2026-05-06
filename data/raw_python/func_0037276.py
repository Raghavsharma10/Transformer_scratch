def usage(self, period='hour'):
        """ Check the number of objects processed and delivered for a given time period

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/usage

            :param period: (optional) time period to measure usage for, can be one of "day", "hour" or "current" (5 minutes), default is hour
            :type period: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.get('usage', params=dict(period=period))