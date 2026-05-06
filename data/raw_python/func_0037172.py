def get(self, historics_id=None, maximum=None, page=None, with_estimate=None):
        """ Get the historics query with the given ID, if no ID is provided then get a list of historics queries.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsget

            :param historics_id: (optional) ID of the query to retrieve
            :type historics_id: str
            :param maximum: (optional) maximum number of queries to recieve (default 20)
            :type maximum: int
            :param page: (optional) page to retrieve for paginated queries
            :type page: int
            :param with_estimate: include estimate of completion time in output
            :type with_estimate: bool
            :param historics_id: playback id of the query
            :type historics_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': historics_id}
        if maximum:
            params['max'] = maximum
        if page:
            params['page'] = page

        params['with_estimate'] = 1 if with_estimate else 0
        return self.request.get('get', params=params)