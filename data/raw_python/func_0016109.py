def delete_event_subscription(self, url):
        """Deregister a callback URL as an event subscriber.

        :param str url: callback URL

        :returns: the deleted event subscription
        :rtype: dict
        """
        params = {'callbackUrl': url}
        response = self._do_request('DELETE', '/v2/eventSubscriptions', params)
        return response.json()