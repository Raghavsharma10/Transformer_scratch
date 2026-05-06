def update(self, historics_id, name):
        """ Update the name of the given Historics query.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsupdate

            :param historics_id: playback id of the job to start
            :type historics_id: str
            :param name: new name of the stream
            :type name: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('update', data=dict(id=historics_id, name=name))