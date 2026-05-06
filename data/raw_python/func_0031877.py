def get_all_client_events(self, client):
        """
        Returns the list of current events for a given client.
        """
        data = self._request('GET', '/events/{}'.format(client))
        return data.json()