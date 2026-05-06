def list(self, identity_id, per_page=20, page=1):
        """ Get a list of tokens

            :param identity_id: The ID of the identity to retrieve tokens for
            :param per_page: The number of results per page returned
            :param page: The page number of the results
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'per_page': per_page, 'page': page}

        return self.request.get(str(identity_id) + '/token', params)