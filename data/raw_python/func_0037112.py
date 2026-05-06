def get(self, identity_id, service):
        """ Get the limit for the given identity and service

            :param identity_id: The ID of the identity to retrieve
            :param service: The service that the limit is linked to
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        return self.request.get(str(identity_id) + '/limit/' + service)