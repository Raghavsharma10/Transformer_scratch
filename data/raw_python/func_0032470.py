def fetch(self):
        """
        Fetch & return a new `FloatingIP` object representing the floating IP's
        current state

        :rtype: FloatingIP
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the floating IP no longer exists)
        """
        api = self.doapi_manager
        return api._floating_ip(api.request(self.url)["floating_ip"])