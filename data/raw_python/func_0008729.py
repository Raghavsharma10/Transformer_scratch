def get_settings(self):
        """
        Returns current settings.

        Only accessible if authenticated as the user.
        """
        url = self._imgur._base_url + "/3/account/{0}/settings".format(self.name)
        return self._imgur._send_request(url)