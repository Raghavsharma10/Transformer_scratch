def is_following(self, login):
        """Check if the authenticated user is following login.

        :param str login: (required), login of the user to check if the
            authenticated user is checking
        :returns: bool
        """
        json = False
        if login:
            url = self._build_url('user', 'following', login)
            json = self._boolean(self._get(url), 204, 404)
        return json