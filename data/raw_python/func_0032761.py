def fetch(self):
        """
        Fetch & return a new `Tag` object representing the tag's current state

        :rtype: Tag
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the tag no longer exists)
        """
        api = self.doapi_manager
        return api._tag(api.request(self.url)["tag"])