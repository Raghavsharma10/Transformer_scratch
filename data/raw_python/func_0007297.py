def get_user(self):
        """Access basic account information."""
        method = 'GET'
        endpoint = '/rest/v1/users/{}'.format(self.client.sauce_username)
        return self.client.request(method, endpoint)