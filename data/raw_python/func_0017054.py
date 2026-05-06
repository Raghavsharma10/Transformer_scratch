def iter_starred(self, sort=None, direction=None, number=-1, etag=None):
        """Iterate over repositories starred by this user.

        .. versionchanged:: 0.5
           Added sort and direction parameters (optional) as per the change in
           GitHub's API.

        :param int number: (optional), number of starred repos to return.
            Default: -1, returns all available repos
        :param str sort: (optional), either 'created' (when the star was
            created) or 'updated' (when the repository was last pushed to)
        :param str direction: (optional), either 'asc' or 'desc'. Default:
            'desc'
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        from .repos import Repository

        params = {'sort': sort, 'direction': direction}
        self._remove_none(params)
        url = self.starred_urlt.expand(owner=None, repo=None)
        return self._iter(int(number), url, Repository, params, etag)