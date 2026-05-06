def get_usage(self, start=None, end=None):
        """Access historical account usage data."""
        method = 'GET'
        endpoint = '/rest/v1/users/{}/usage'.format(self.client.sauce_username)
        data = {}
        if start:
            data['start'] = start
        if end:
            data['end'] = end
        if data:
            endpoint = '?'.join([endpoint, urlencode(data)])
        return self.client.request(method, endpoint)