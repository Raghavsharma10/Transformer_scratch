def get_stored_files(self):
        """Check which files are in your temporary storage."""
        method = 'GET'
        endpoint = '/rest/v1/storage/{}'.format(self.client.sauce_username)
        return self.client.request(method, endpoint)