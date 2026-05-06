def unfollow(self, login):
        """Make the authenticated user stop following login

        :param str login: (required)
        :returns: bool
        """
        resp = False
        if login:
            url = self._build_url('user', 'following', login)
            resp = self._boolean(self._delete(url), 204, 404)
        return resp