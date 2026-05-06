def get_event(self, client, check):
        """
        Returns an event for a given client & check name.
        """
        data = self._request('GET', '/events/{}/{}'.format(client, check))
        return data.json()