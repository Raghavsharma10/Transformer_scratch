def get_siblings(self):
        """Get a list of sibling accounts associated with provided account."""
        method = 'GET'
        endpoint = '/rest/v1.1/users/{}/siblings'.format(
            self.client.sauce_username)
        return self.client.request(method, endpoint)