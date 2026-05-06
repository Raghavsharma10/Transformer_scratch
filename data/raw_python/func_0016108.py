def create_event_subscription(self, url):
        """Register a callback URL as an event subscriber.

        :param str url: callback URL

        :returns: the created event subscription
        :rtype: dict
        """
        params = {'callbackUrl': url}
        response = self._do_request('POST', '/v2/eventSubscriptions', params)
        return response.json()