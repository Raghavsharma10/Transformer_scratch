def get(self, preview_id):
        """ Retrieve a Historics preview job.

            Warning: previews expire after 24 hours.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/previewget

            :param preview_id: historics preview job hash of the job to retrieve
            :type preview_id: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self.request.get('get', params=dict(id=preview_id))