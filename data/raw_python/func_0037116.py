def delete(self, identity_id, service):
        """ Delete the limit for the given identity and service

            :param identity_id: The ID of the identity to retrieve
            :param service: The service that the token is linked to
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        return self.request.delete(str(identity_id) + '/limit/' + service)