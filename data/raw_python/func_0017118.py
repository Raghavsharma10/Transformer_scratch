def iter_statuses(self, sha, number=-1, etag=None):
        """Iterates over the statuses for a specific SHA.

        :param str sha: SHA of the commit to list the statuses of
        :param int number: (optional), return up to number statuses. Default:
            -1 returns all available statuses.
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of :class:`Status <github3.repos.status.Status>`
        """
        url = ''
        if sha:
            url = self._build_url('statuses', sha, base_url=self._api)
        return self._iter(int(number), url, Status, etag=etag)