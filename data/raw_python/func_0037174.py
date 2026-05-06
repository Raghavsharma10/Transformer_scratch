def resume(self, historics_id):
        """ Resume a paused Historics query.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsresume

            :param historics_id: id of the job to resume
            :type historics_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('resume', data=dict(id=historics_id))