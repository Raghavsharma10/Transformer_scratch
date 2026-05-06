def get_platforms(self, automation_api='all'):
        """Get a list of objects describing all the OS and browser platforms
        currently supported on Sauce Labs."""
        method = 'GET'
        endpoint = '/rest/v1/info/platforms/{}'.format(automation_api)
        return self.client.request(method, endpoint)