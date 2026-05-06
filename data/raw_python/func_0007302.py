def get_subaccount_info(self):
        """Get information about a sub account."""
        method = 'GET'
        endpoint = '/rest/v1/users/{}/subaccounts'.format(
            self.client.sauce_username)
        return self.client.request(method, endpoint)