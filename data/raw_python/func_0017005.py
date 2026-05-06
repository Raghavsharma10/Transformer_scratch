def iter_all_repos(self, number=-1, since=None, etag=None, per_page=None):
        """Iterate over every repository in the order they were created.

        :param int number: (optional), number of repositories to return.
            Default: -1, returns all of them
        :param int since: (optional), last repository id seen (allows
            restarting this iteration)
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :param int per_page: (optional), number of repositories to list per
            request
        :returns: generator of :class:`Repository <github3.repos.Repository>`
        """
        url = self._build_url('repositories')
        return self._iter(int(number), url, Repository,
                          params={'since': since, 'per_page': per_page},
                          etag=etag)