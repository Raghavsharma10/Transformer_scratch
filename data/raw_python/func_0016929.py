def invite(self, username):
        """Invite the user to join this team.

        This returns a dictionary like so::

            {'state': 'pending', 'url': 'https://api.github.com/teams/...'}

        :param str username: (required), user to invite to join this team.
        :returns: dictionary
        """
        url = self._build_url('memberships', username, base_url=self._api)
        return self._json(self._put(url), 200)