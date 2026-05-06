def iter_repos(self, type=None, sort=None, direction=None, number=-1,
                   etag=None):
        """List public repositories for the authenticated user.

        .. versionchanged:: 0.6
           Removed the login parameter for correctness. Use iter_user_repos
           instead

        :param str type: (optional), accepted values:
            ('all', 'owner', 'public', 'private', 'member')
            API default: 'all'
        :param str sort: (optional), accepted values:
            ('created', 'updated', 'pushed', 'full_name')
            API default: 'created'
        :param str direction: (optional), accepted values:
            ('asc', 'desc'), API default: 'asc' when using 'full_name',
            'desc' otherwise
        :param int number: (optional), number of repositories to return.
            Default: -1 returns all repositories
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
            objects
        """
        url = self._build_url('user', 'repos')

        params = {}
        if type in ('all', 'owner', 'public', 'private', 'member'):
            params.update(type=type)
        if sort in ('created', 'updated', 'pushed', 'full_name'):
            params.update(sort=sort)
        if direction in ('asc', 'desc'):
            params.update(direction=direction)

        return self._iter(int(number), url, Repository, params, etag)