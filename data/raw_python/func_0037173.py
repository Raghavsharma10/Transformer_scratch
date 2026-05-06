def pause(self, historics_id, reason=""):
        """ Pause an existing Historics query.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/historicspause

            :param historics_id: id of the job to pause
            :type historics_id: str
            :param reason: optional reason for pausing it
            :type reason: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {"id": historics_id}
        if reason != "":
            params["reason"] = reason
        return self.request.post('pause', data=params)