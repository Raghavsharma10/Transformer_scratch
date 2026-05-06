def fetch(self):
        """
        Fetch & return a new `SSHKey` object representing the SSH key's current
        state

        :rtype: SSHKey
        :raises DOAPIError: if the API endpoint replies with an error (e.g., if
            the SSH key no longer exists)
        """
        api = self.doapi_manager
        return api._ssh_key(api.request(self.url)["ssh_key"])