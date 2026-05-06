def membership_for(self, username):
        """Retrieve the membership information for the user.

        :param str username: (required), name of the user
        :returns: dictionary
        """
        url = self._build_url('memberships', username, base_url=self._api)
        json = self._json(self._get(url), 200)
        return json or {}