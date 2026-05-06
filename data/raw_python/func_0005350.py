def remove_api_key(self):
        """
        Removes the user's existing API key, if present, and sets the current instance's 'api_key'
        attribute to the empty string.

        Returns:
            `NoneType`: None.
        """
        url = self.record_url + "/remove_api_key"
        res = requests.patch(url=url, headers=HEADERS, verify=False)
        res.raise_for_status()
        self.api_key = ""