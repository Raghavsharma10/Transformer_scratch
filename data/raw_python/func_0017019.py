def iter_subscriptions(self, login=None, number=-1, etag=None):
        """Iterate over repositories subscribed to by ``login`` or the
        authenticated user.

        :param str login: (optional), name of user whose subscriptions you want
            to see
        :param int number: (optional), number of repositories to return.
            Default: -1 returns all repositories
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        if login:
            return self.user(login).iter_subscriptions()

        url = self._build_url('user', 'subscriptions')
        return self._iter(int(number), url, Repository, etag=etag)