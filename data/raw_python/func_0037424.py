def list(self, label=None, per_page=20, page=1):
        """ Get a list of identities that have been created

            :param per_page: The number of results per page returned
            :type per_page: int
            :param page: The page number of the results
            :type page: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {'per_page': per_page, 'page': page}

        if label:
            params['label'] = label

        return self.request.get('', params)