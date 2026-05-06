def iter_starred(self, login=None, sort=None, direction=None, number=-1,
                     etag=None):
        """Iterate over repositories starred by ``login`` or the authenticated
        user.

        .. versionchanged:: 0.5
           Added sort and direction parameters (optional) as per the change in
           GitHub's API.

        :param str login: (optional), name of user whose stars you want to see
        :param str sort: (optional), either 'created' (when the star was
            created) or 'updated' (when the repository was last pushed to)
        :param str direction: (optional), either 'asc' or 'desc'. Default:
            'desc'
        :param int number: (optional), number of repositories to return.
            Default: -1 returns all repositories
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        if login:
            return self.user(login).iter_starred(sort, direction)

        params = {'sort': sort, 'direction': direction}
        self._remove_none(params)
        url = self._build_url('user', 'starred')
        return self._iter(int(number), url, Repository, params, etag)