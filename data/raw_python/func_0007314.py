def get_tunnels(self):
        """Retrieves all running tunnels for a specific user."""
        method = 'GET'
        endpoint = '/rest/v1/{}/tunnels'.format(self.client.sauce_username)
        return self.client.request(method, endpoint)