def iter_followers(self, number=-1, etag=None):
        """Iterate over the followers of this user.

        :param int number: (optional), number of followers to return. Default:
            -1 returns all available
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`User <User>`\ s
        """
        url = self._build_url('followers', base_url=self._api)
        return self._iter(int(number), url, User, etag=etag)