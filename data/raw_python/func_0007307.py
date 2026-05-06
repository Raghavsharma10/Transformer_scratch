def get_jobs(self, full=None, limit=None, skip=None, start=None, end=None,
                 output_format=None):
        """List jobs belonging to a specific user."""
        method = 'GET'
        endpoint = '/rest/v1/{}/jobs'.format(self.client.sauce_username)
        data = {}
        if full is not None:
            data['full'] = full
        if limit is not None:
            data['limit'] = limit
        if skip is not None:
            data['skip'] = skip
        if start is not None:
            data['from'] = start
        if end is not None:
            data['to'] = end
        if output_format is not None:
            data['format'] = output_format
        if data:
            endpoint = '?'.join([endpoint, urlencode(data)])
        return self.client.request(method, endpoint)