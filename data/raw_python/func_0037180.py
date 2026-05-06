def log(self, source_id, page=None, per_page=None):
        """ Get the log for a specific Managed Source.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourcelog

            :param source_id: target Source ID
            :type source_id: str
            :param page: (optional) page number for pagination
            :type page: int
            :param per_page: (optional) number of items per page, default 20
            :type per_page: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': source_id}
        if page:
            params['page'] = page
        if per_page:
            params['per_page'] = per_page

        return self.request.get('log', params=params)