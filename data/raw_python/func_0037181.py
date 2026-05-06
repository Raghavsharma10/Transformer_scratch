def get(self, source_id=None, source_type=None, page=None, per_page=None):
        """ Get a specific managed source or a list of them.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/sourceget

            :param source_id: (optional) target Source ID
            :type source_id: str
            :param source_type: (optional) data source name e.g. facebook_page, googleplus, instagram, yammer
            :type source_type: str
            :param page: (optional) page number for pagination, default 1
            :type page: int
            :param per_page: (optional) number of items per page, default 20
            :type per_page: int
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {}
        if source_type:
            params['source_type'] = source_type
        if source_id:
            params['id'] = source_id
        if page:
            params['page'] = page
        if per_page:
            params['per_page'] = per_page

        return self.request.get('get', params=params)