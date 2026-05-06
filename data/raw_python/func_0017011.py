def iter_following(self, login=None, number=-1, etag=None):
        """If login is provided, iterate over a generator of users being
        followed by login; otherwise return a generator of people followed by
        the authenticated user.

        :param str login: (optional), login of the user to check
        :param int number: (optional), number of people to return. Default: -1
            returns all people you follow
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`User <github3.users.User>`\ s
        """
        if login:
            return self.user(login).iter_following()
        return self._iter_follow('following', int(number), etag=etag)