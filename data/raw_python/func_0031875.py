def get_client_data(self, client):
        """
        Returns a client.
        """
        data = self._request('GET', '/clients/{}'.format(client))
        return data.json()