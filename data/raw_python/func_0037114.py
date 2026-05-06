def create(self, identity_id, service, total_allowance=None, analyze_queries=None):
        """ Create the limit

            :param identity_id: The ID of the identity to retrieve
            :param service: The service that the token is linked to
            :param total_allowance: The total allowance for this token's limit
            :param analyze_queries: The number of analyze calls
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'service': service}

        if total_allowance is not None:
            params['total_allowance'] = total_allowance
        if analyze_queries is not None:
            params['analyze_queries'] = analyze_queries

        return self.request.post(str(identity_id) + '/limit/', params)