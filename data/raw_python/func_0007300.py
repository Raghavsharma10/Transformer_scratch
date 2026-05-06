def get_subaccounts(self):
        """Get a list of sub accounts associated with a parent account."""
        method = 'GET'
        endpoint = '/rest/v1/users/{}/list-subaccounts'.format(
            self.client.sauce_username)
        return self.client.request(method, endpoint)