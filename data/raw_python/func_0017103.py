def iter_contributors(self, anon=False, number=-1, etag=None):
        """Iterate over the contributors to this repository.

        :param bool anon: (optional), True lists anonymous contributors as
            well
        :param int number: (optional), number of contributors to return.
            Default: -1 returns all contributors
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`User <github3.users.User>`\ s
        """
        url = self._build_url('contributors', base_url=self._api)
        params = {}
        if anon:
            params = {'anon': True}
        return self._iter(int(number), url, User, params, etag)