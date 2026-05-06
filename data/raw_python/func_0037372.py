def log(self, subscription_id=None, page=None, per_page=None, order_by=None, order_dir=None):
        """ Retrieve any messages that have been logged for your subscriptions.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushlog

            :param subscription_id: optional id of an existing Push Subscription, restricts logs to a given subscription if supplied.
            :type subscription_id: str
            :param page: optional page number for pagination
            :type page: int
            :param per_page: optional number of items per page, default 20
            :type per_page: int
            :param order_by: field to order by, default request_time
            :type order_by: str
            :param order_dir: direction to order by, asc or desc, default desc
            :type order_dir: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {}
        if subscription_id:
            params['id'] = subscription_id
        if page:
            params['page'] = page
        if per_page:
            params['per_page'] = per_page
        if order_by:
            params['order_by'] = order_by
        if order_dir:
            params['order_dir'] = order_dir

        return self.request.get('log', params=params)