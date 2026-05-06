def get_clients(self, limit=None, offset=None):
        """
        Returns a list of clients.
        """
        data = {}
        if limit:
            data['limit'] = limit
        if offset:
            data['offset'] = offset
        result = self._request('GET', '/clients', data=json.dumps(data))
        return result.json()