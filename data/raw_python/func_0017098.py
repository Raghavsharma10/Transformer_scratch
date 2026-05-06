def iter_branches(self, number=-1, etag=None):
        """Iterate over the branches in this repository.

        :param int number: (optional), number of branches to return. Default:
            -1 returns all branches
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of
            :class:`Branch <github3.repos.branch.Branch>`\ es
        """
        url = self._build_url('branches', base_url=self._api)
        return self._iter(int(number), url, Branch, etag=etag)