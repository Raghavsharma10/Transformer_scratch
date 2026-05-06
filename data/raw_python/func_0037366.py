def create_from_historics(self, historics_id, name, output_type, output_params, initial_status=None, start=None,
                              end=None):
        """ Create a new push subscription using the given Historic ID.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushcreate

            :param historics_id: The ID of a Historics query
            :type historics_id: str
            :param name: The name to give the newly created subscription
            :type name: str
            :param output_type: One of the supported output types e.g. s3
            :type output_type: str
            :param output_params: set of parameters required for the given output type, see dev.datasift.com
            :type output_params: dict
            :param initial_status: The initial status of the subscription, active, paused or waiting_for_start
            :type initial_status: str
            :param start: Optionally specifies when the subscription should start
            :type start: int
            :param end: Optionally specifies when the subscription should end
            :type end: int
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        return self._create(False, historics_id, name, output_type, output_params, initial_status, start, end)