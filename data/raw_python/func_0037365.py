def create_from_hash(self, stream, name, output_type, output_params,
                         initial_status=None, start=None, end=None):
        """ Create a new push subscription using a live stream.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushcreate

            :param stream: The hash of a DataSift stream.
            :type stream: str
            :param name: The name to give the newly created subscription
            :type name: str
            :param output_type: One of the supported output types e.g. s3
            :type output_type: str
            :param output_params: The set of parameters required for the given output type
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
        return self._create(True, stream, name, output_type, output_params, initial_status, start, end)