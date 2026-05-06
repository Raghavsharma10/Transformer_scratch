def iter_repos(self, type='', number=-1, etag=None):
        """Iterate over repos for this organization.

        :param str type: (optional), accepted values:
            ('all', 'public', 'member', 'private', 'forks', 'sources'), API
            default: 'all'
        :param int number: (optional), number of members to return. Default:
            -1 will return all available.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        url = self._build_url('repos', base_url=self._api)
        params = {}
        if type in ('all', 'public', 'member', 'private', 'forks', 'sources'):
            params['type'] = type
        return self._iter(int(number), url, Repository, params, etag)