def stop(self, subscription_id):
        """ Stop the given subscription from running.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushstop

            :param subscription_id: id of an existing Push Subscription.
            :type subscription_id: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`

        """
        return self.request.post('stop', data=dict(id=subscription_id))