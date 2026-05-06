def iter_followers(self, login=None, number=-1, etag=None):
        """If login is provided, iterate over a generator of followers of that
        login name; otherwise return a generator of followers of the
        authenticated user.

        :param str login: (optional), login of the user to check
        :param int number: (optional), number of followers to return. Default:
            -1 returns all followers
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`User <github3.users.User>`\ s
        """
        if login:
            return self.user(login).iter_followers()
        return self._iter_follow('followers', int(number), etag=etag)