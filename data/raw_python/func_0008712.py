def get_user(self, username):
        """
        Return a User object for this username.

        :param username: The name of the user we want more information about.
        """
        url = self._base_url + "/3/account/{0}".format(username)
        json = self._send_request(url)
        return User(json, self)