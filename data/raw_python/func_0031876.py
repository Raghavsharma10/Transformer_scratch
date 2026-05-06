def get_client_history(self, client):
        """
        Returns the history for a client.
        """
        data = self._request('GET', '/clients/{}/history'.format(client))
        return data.json()