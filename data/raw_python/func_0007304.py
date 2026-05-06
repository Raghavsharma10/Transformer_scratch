def get_activity(self):
        """Check account concurrency limits."""
        method = 'GET'
        endpoint = '/rest/v1/{}/activity'.format(self.client.sauce_username)
        return self.client.request(method, endpoint)