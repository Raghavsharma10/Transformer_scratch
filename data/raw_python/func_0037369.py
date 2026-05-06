def update(self, subscription_id, output_params, name=None):
        """ Update the name or output parameters for an existing Subscription.

            Uses API documented at http://dev.datasift.com/docs/api/rest-api/endpoints/pushupdate

            :param subscription_id: id of an existing Push Subscription.
            :type subscription_id: str
            :param output_params: new output parameters for the subscription, see dev.datasift.com
            :type output_params: dict
            :param name: optional new name for the Subscription
            :type name: str
            :returns: dict with extra response data
            :rtype: :class:`~datasift.request.DictResponse`
            :raises: :class:`~datasift.exceptions.DataSiftApiException`, :class:`requests.exceptions.HTTPError`
        """
        params = {'id': subscription_id, 'output_params': output_params}
        if name:
            params['name'] = name
        return self.request.post('update', params)