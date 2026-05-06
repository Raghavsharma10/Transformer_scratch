def fetch(self):
        """
        Fetch & return a new `Domain` object representing the domain's current
        state

        :rtype: Domain
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the domain no longer exists)
        """
        api = self.doapi_manager
        return api._domain(api.request(self.url)["domain"])