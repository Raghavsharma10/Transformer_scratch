def resume(self, subscription_id):
        """ Resume a previously paused Subscription.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushresume

            :param subscription_id: id of an existing Push Subscription.
            :type subscription_id: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`

        """
        return self.request.post('resume', data=dict(id=subscription_id))