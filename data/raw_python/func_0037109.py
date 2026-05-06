def list(self, page=None, per_page=None, order_by='created_at',
             order_dir='DESC', service='facebook'):

        """ List pylon recordings

            :param page: page number for pagination
            :type page: int
            :param per_page: number of items per page, default 20
            :type per_page: int
            :param order_by: field to order by, default request_time
            :type order_by: str
            :param order_dir: direction to order by, asc or desc, default desc
            :type order_dir: str
            :param service: The service for this API call (facebook, etc)
            :type service: str
            :return: dict of REST API output with headers attached
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`,
                :class:`requests.exceptions.HTTPError`
        """

        params = {}

        if page:
            params['page'] = page
        if per_page:
            params['per_page'] = per_page
        if order_by:
            params['order_by'] = order_by
        if order_dir:
            params['order_dir'] = order_dir

        return self.request.get(service + '/get', params)