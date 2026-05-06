def update(self, identity_id, service, token=None):
        """ Update the token

            :param identity_id: The ID of the identity to retrieve
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {}

        if token:
            params['token'] = token

        return self.request.put(str(identity_id) + '/token/' + service, params)