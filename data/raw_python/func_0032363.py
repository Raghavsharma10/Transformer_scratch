def fetch(self):
        """
        Fetch & return a new `Image` object representing the image's current
        state

        :rtype: Image
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the image no longer exists)
        """
        api = self.doapi_manager
        return api._image(api.request(self.url)["image"])