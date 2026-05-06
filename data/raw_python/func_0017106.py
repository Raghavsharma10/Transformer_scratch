def iter_forks(self, sort='', number=-1, etag=None):
        """Iterate over forks of this repository.

        :param str sort: (optional), accepted values:
            ('newest', 'oldest', 'watchers'), API default: 'newest'
        :param int number: (optional), number of forks to return. Default: -1
            returns all forks
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Repository <Repository>`
        """
        url = self._build_url('forks', base_url=self._api)
        params = {}
        if sort in ('newest', 'oldest', 'watchers'):
            params = {'sort': sort}
        return self._iter(int(number), url, Repository, params, etag)