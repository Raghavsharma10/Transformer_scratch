def delete(self, historics_id):
        """ Delete one specified playback query. If the query is currently running, stop it.

            status_code is set to 204 on success

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicsdelete

            :param historics_id: playback id of the query to delete
            :type historics_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.post('delete', data=dict(id=historics_id))