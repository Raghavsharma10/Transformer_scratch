def stop(self, historics_id, reason=''):
        """ Stop an existing Historics query.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsstop

            :param historics_id: playback id of the job to stop
            :type historics_id: str
            :param reason: optional reason for stopping the job
            :type reason: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('stop', data=dict(id=historics_id, reason=reason))