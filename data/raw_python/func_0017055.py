def iter_subscriptions(self, number=-1, etag=None):
        """Iterate over repositories subscribed to by this user.

        :param int number: (optional), number of subscriptions to return.
            Default: -1, returns all available
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        from .repos import Repository
        url = self._build_url('subscriptions', base_url=self._api)
        return self._iter(int(number), url, Repository, etag=etag)