def create(self, identity_id, service, token):
        """ Create the token

            :param identity_id: The ID of the identity to retrieve
            :param service: The service that the token is linked to
            :param token: The token provided by the the service
            :param expires_at: Set an expiry for this token
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'service': service, 'token': token}

        return self.request.post(str(identity_id) + '/token', params)