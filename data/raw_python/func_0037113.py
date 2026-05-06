def list(self, service, per_page=20, page=1):
        """ Get a list of limits for the given service

            :param service: The service that the limit is linked to
            :param per_page: The number of results per page returned
            :param page: The page number of the results
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'per_page': per_page, 'page': page}

        return self.request.get('limit/' + service, params)