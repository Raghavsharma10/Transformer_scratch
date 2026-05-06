def fetch(self):
        """
        Fetch & return a new `Droplet` object representing the droplet's
        current state

        :rtype: Droplet
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the droplet no longer exists)
        """
        api = self.doapi_manager
        return api._droplet(api.request(self.url)["droplet"])