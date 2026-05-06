def add(self, source_id, auth, validate=True):
        """ Add one or more sets of authorization credentials to a Managed Source

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourceauthadd

            :param source_id: target Source ID
            :type source_id: str
            :param auth: An array of the source-specific authorization credential sets that you're adding.
            :type auth: array of strings
            :param validate: Allows you to suppress the validation of the authorization credentials, defaults to true.
            :type validate: bool
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': source_id, 'auth': auth, 'validate': validate}
        return self.request.post('add', params)