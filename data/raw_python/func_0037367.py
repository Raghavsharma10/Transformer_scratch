def pause(self, subscription_id):
        """ Pause a Subscription and buffer the data for up to one hour.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushpause

            :param subscription_id: id of an existing Push Subscription.
            :type subscription_id: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`

        """
        return self.request.post('pause', data=dict(id=subscription_id))