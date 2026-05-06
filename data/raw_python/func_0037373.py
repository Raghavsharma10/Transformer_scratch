def get(self, subscription_id=None, stream=None, historics_id=None,
            page=None, per_page=None, order_by=None, order_dir=None,
            include_finished=None):
        """ Show details of the Subscriptions belonging to this user.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushget

            :param subscription_id: optional id of an existing Push Subscription
            :type subscription_id: str
            :param hash: optional hash of a live stream
            :type hash: str
            :param playback_id: optional playback id of a Historics query
            :type playback_id: str
            :param page: optional page number for pagination
            :type page: int
            :param per_page: optional number of items per page, default 20
            :type per_page: int
            :param order_by: field to order by, default request_time
            :type order_by: str
            :param order_dir: direction to order by, asc or desc, default desc
            :type order_dir: str
            :param include_finished: boolean indicating if finished Subscriptions for Historics should be included
            :type include_finished: bool
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {}
        if subscription_id:
            params['id'] = subscription_id
        if stream:
            params['hash'] = stream
        if historics_id:
            params['historics_id'] = historics_id
        if page:
            params['page'] = page
        if per_page:
            params['per_page'] = per_page
        if order_by:
            params['order_by'] = order_by
        if order_dir:
            params['order_dir'] = order_dir
        if include_finished:
            params['include_finished'] = 1 if include_finished else 0

        return self.request.get('get', params=params)