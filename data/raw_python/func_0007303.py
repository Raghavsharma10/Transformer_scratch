def change_access_key(self):
        """Change access key of your account."""
        method = 'POST'
        endpoint = '/rest/v1/users/{}/accesskey/change'.format(
            self.client.sauce_username)
        return self.client.request(method, endpoint)