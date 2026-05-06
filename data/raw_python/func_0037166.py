def start(self, historics_id):
        """ Start the historics job with the given ID.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsstart

            :param historics_id: hash of the job to start
            :type historics_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('start', data=dict(id=historics_id))