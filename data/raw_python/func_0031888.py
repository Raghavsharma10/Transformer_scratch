def get_result(self, client, check):
        """
        Returns an event for a given client & result name.
        """
        data = self._request('GET', '/results/{}/{}'.format(client, check))
        return data.json()