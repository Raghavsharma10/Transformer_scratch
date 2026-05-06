def get_for(self, historics_id, with_estimate=None):
        """ Get the historic query for the given ID

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsget

            :param historics_id: playback id of the query
            :type historics_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.get(historics_id, maximum=None, page=None, with_estimate=with_estimate)