def follow(self, login):
        """Make the authenticated user follow login.

        :param str login: (required), user to follow
        :returns: bool
        """
        resp = False
        if login:
            url = self._build_url('user', 'following', login)
            resp = self._boolean(self._put(url), 204, 404)
        return resp