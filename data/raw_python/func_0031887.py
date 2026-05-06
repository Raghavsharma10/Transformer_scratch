def get_results(self, client):
        """
        Returns a result.
        """
        data = self._request('GET', '/results/{}'.format(client))
        return data.json()